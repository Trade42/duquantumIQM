"""Rotated surface-code memory circuits, with and without ancilla reset (challenge criterion 1).

``memory_circuit(distance, rounds, basis, no_reset, noise)`` builds the memory experiment of
the M1 contract (``orchestration/briefs/_m1-contract.md``, section 1) as a ``stim.Circuit``.
The geometry, the CX schedule and the detectors are written here, not taken from
``stim.Circuit.generated``. With ``no_reset=False`` and the challenge's noise, the Z-basis
circuit is, instruction for instruction, the flattened ``generated("surface_code:rotated_memory_z")``,
and the X-basis circuit has the same detector error model as ``rotated_memory_x``
(``tests/test_surface.py``, ``scripts/surface_check.py``).

Geometry (Stim's rotated-code convention)
-----------------------------------------
Qubits sit on the integer grid 0 <= x, y <= 2d.

* Data qubits are at odd (x, y): d^2 of them.
* Measure qubits are at even (x, y): X-type where x + y = 2 (mod 4), Z-type where
  x + y = 0 (mod 4). Every interior site (2 <= x, y <= 2d - 2) is used; on the boundary only
  the X-type sites of y = 0 and y = 2d and the Z-type sites of x = 0 and x = 2d, never a
  corner. That is (d^2 - 1)/2 of each type.
* The stabilizer at (x, y) acts on the data at (x +- 1, y +- 1): weight 4 inside, 2 on the
  boundary.
* Qubit index q = x + (2d + 1) * floor(y / 2), as Stim numbers them. Every qubit gets a
  ``QUBIT_COORDS(x, y)``, so data and measure qubits are told apart by coordinate parity
  (data: x and y both odd).
* Z_L is Z on the data row y = 1, X_L is X on the data column x = 1.

Schedule
--------
A round is: H on the X-type ancillas; four CX layers; H on the X-type ancillas; measure every
ancilla. In layer i, an X-type ancilla is the control of a CX onto its data neighbour at offset
``X_ORDER[i]``, and a Z-type ancilla is the target of a CX from its neighbour at ``Z_ORDER[i]``
(an ancilla with no neighbour there idles). The X order traces an "N" and the Z order a "Z",
so a hook error (an ancilla fault halfway through, spread onto two data qubits) lies
perpendicular to the logical it could shorten, and the circuit-level distance stays d. This
is the order ``stim.Circuit.generated`` uses.

Detectors without reset
-----------------------
For one ancilla, let s_k be the outcome of its stabilizer measurement in round k
(k = 1 .. R), and m_k the bit its measurement records.

* With reset (``MR``) the ancilla starts every round in |0>, so m_k = s_k.
* Without reset (``M``) it starts round k in |m_(k-1)>, and in |0> for k = 1:
  - Z-type: the CXs add the Z parity of the support onto it, so m_k = m_(k-1) + s_k;
  - X-type: H|m_(k-1)> is the X eigenstate of eigenvalue (-1)^m_(k-1), and the CXs, with the
    ancilla as control, map X_a to X_a X_S, so the X measurement also gives
    m_k = m_(k-1) + s_k.

  So m_k = s_1 + ... + s_k and s_k = m_k + m_(k-1), with m_0 = m_(-1) = 0 (sums mod 2).

A detector is a stabilizer's change between rounds, D_k = s_k + s_(k-1). Which stabilizers
have one in round 1 depends on the basis:

* the basis type (Z-type in the Z basis, X-type in the X basis) is deterministic from the
  start: the data begin in |0...0> (|+...+>), their +1 eigenstate, so s_0 = 0;
* the other type starts random: s_1 is a coin flip, so it has no round-1 detector.

=============  ================  =======================  ===============
detector       with reset (MR)   without reset (M)        ancillas
=============  ================  =======================  ===============
round 1        m_1               m_1                      basis type
round 2        m_2 + m_1         m_2  (= m_2 + m_0)       all
round k >= 3   m_k + m_(k-1)     m_k + m_(k-2)            all
final          P + m_R           P + m_R + m_(R-1)        basis type
=============  ================  =======================  ===============

P is the parity of the final data measurement over the stabilizer's support, and m_0 = 0 when
R = 1. The round-2 detector of the other type is m_2 alone although m_1 is random: the
ancilla has accumulated s_1 twice, m_2 = s_1 + s_2 = 0. Every prepared state is a +1
eigenstate and every ancilla starts in |0>, so in the noiseless circuit every detector's raw
parity, and the observable's, is 0 on every shot. The final data measurement gives only the
basis-type stabilizers, which is why the other type has no final detector.

Counts: (R + 1)(d^2 - 1)/2 basis-type detectors and (R - 1)(d^2 - 1)/2 of the other type,
R (d^2 - 1) in all; R (d^2 - 1) + d^2 measurements; one observable.

Without reset, a flip of an ancilla's state after its round-k measurement flips every later
m, so it flips s_(k+1) alone and fires D_(k+1) and D_(k+2): exactly what a reset error does
with ``MR``. A flip just before the measurement fires D_k and D_(k+1), in both variants. So
at equal rates, a flip of the live ancilla during the measurement window (``idle``, below)
and a reset error give the same detector error model. A record-only readout error (the bit
flips, the state does not) is different: without reset it flips m_k alone, so it fires D_k and
D_(k+2), two rounds apart (and the final detector instead of D_(k+2) when k >= R - 1). With
``MR`` it fires D_k and D_(k+1), like any flip before the measurement. ``record_flip_probability``
models it (below).

Noise
-----
``NoiseModel`` places each channel where ``stim.Circuit.generated`` puts it:

* ``after_reset_flip_probability``: ``X_ERROR`` after every ``R``, and after every ``MR``;
* ``before_measure_flip_probability``: ``X_ERROR`` before every ``M`` and ``MR``;
* ``after_clifford_depolarization``: ``DEPOLARIZE1`` after every ancilla ``H``, ``DEPOLARIZE2``
  after every ``CX``;
* ``before_round_data_depolarization``: ``DEPOLARIZE1`` on every data qubit at the start of
  every round.

In the X basis, the data's ``H`` after ``R`` and before the final ``M`` are the preparation and
the measurement, like Stim's ``RX`` and ``MX``: they carry no depolarization, and the flips
around them are the same (``X_ERROR`` after ``R``, then ``H``, is ``Z_ERROR`` after ``RX``).

``record_flip_probability`` is ours and is 0 unless set. It is the readout's assignment error:
the recorded bit flips, the qubit does not. It writes ``M(p)`` on every ancilla and data
readout (the contract's record-only flip). With ``MR`` the reset follows at once, so a record
flip and a state flip are the same thing: it writes ``X_ERROR(p)`` before the ``MR`` instead,
since the contract lists no ``MR(p)``. Per-qubit rates split an ``M`` into consecutive runs of
equal rate, so the measurement order does not change. ``final_record_flip_probability`` is the
same flip for the final readout layer (the last round's ancillas and the data), which IQM compiles
to a different readout (``measure_fidelity``); unset, it is ``record_flip_probability``.

``idle_during_measurement`` is ours and is 0 unless set. It is the decay of the qubits that
wait while the ancillas are read out: ``DEPOLARIZE1`` on every data qubit after each ancilla
measurement, in both variants, and ``X_ERROR`` on every ancilla after its ``M`` without reset.
A Z error does nothing to an ancilla in |m>. With ``MR`` the reset wipes the ancilla, so it
gets no idle channel. ``idle_pauli_x``, ``idle_pauli_y`` and ``idle_pauli_z`` are the biased
version on the data: ``PAULI_CHANNEL_1(x, y, z)`` at the same place, written after the
``DEPOLARIZE1`` when both are set. Stim has no amplitude damping: these Pauli channels stand in
for T1 and T2 decay. There is no idle after the last round: its ancillas are read in the same
layer as the data, so nothing waits (``idle_after_last_round=True`` puts it back, only to
rebuild a decoder made before that fix).

Every rate can be set per qubit (``qubit_rates``) and per CX pair (``pair_rates``), for
calibrated noise.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields

import stim

__all__ = [
    "NoiseModel",
    "SurfaceGeometry",
    "X_ORDER",
    "Z_ORDER",
    "geometry",
    "memory_circuit",
]

#: Data offsets (dx, dy) from an X-type ancilla, one per CX layer (the ancilla is the control).
X_ORDER = ((1, 1), (-1, 1), (1, -1), (-1, -1))
#: Data offsets (dx, dy) from a Z-type ancilla, one per CX layer (the ancilla is the target).
Z_ORDER = ((1, 1), (1, -1), (-1, 1), (-1, -1))


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurfaceGeometry:
    """The rotated distance-d code laid out in Stim's coordinates.

    ``x_ancillas`` and ``z_ancillas`` are in coordinate order (x, then y), the order the CX
    layers and the first- and final-round detectors list them in. ``data`` and ``ancillas``
    are in ascending qubit index, the measurement order.
    """

    distance: int
    coords: Mapping[int, tuple[int, int]]
    data: tuple[int, ...]
    ancillas: tuple[int, ...]
    x_ancillas: tuple[int, ...]
    z_ancillas: tuple[int, ...]
    support: Mapping[int, tuple[int, ...]]
    cx_layers: tuple[tuple[tuple[int, int], ...], ...]
    logical_z: tuple[int, ...]
    logical_x: tuple[int, ...]

    def basis_ancillas(self, basis: str) -> tuple[int, ...]:
        """The ancillas whose stabilizers are deterministic in a ``basis`` memory experiment."""
        return self.z_ancillas if basis == "Z" else self.x_ancillas

    def logical(self, basis: str) -> tuple[int, ...]:
        return self.logical_z if basis == "Z" else self.logical_x


def geometry(distance: int) -> SurfaceGeometry:
    """The qubits, stabilizers, CX layers and logicals of the rotated distance-``distance`` code."""
    _check_distance(distance)
    d = distance
    n = 2 * d

    def index(x: int, y: int) -> int:
        return x + (2 * d + 1) * (y // 2)

    data_xy = [(x, y) for x in range(1, n, 2) for y in range(1, n, 2)]
    x_xy, z_xy = [], []
    for x in range(0, n + 1, 2):
        for y in range(0, n + 1, 2):
            is_x = (x + y) % 4 == 2
            on_x_edge, on_y_edge = x in (0, n), y in (0, n)
            if on_x_edge and on_y_edge:
                continue  # a corner touches one data qubit
            if on_y_edge and not is_x:
                continue  # top and bottom edges hold only X-type
            if on_x_edge and is_x:
                continue  # left and right edges hold only Z-type
            (x_xy if is_x else z_xy).append((x, y))
    # coordinate order: x, then y (the loops above already produce it)

    data_set = set(data_xy)
    coords = {index(*p): p for p in data_xy + x_xy + z_xy}
    support = {
        index(x, y): tuple(sorted(index(x + dx, y + dy) for dx in (-1, 1) for dy in (-1, 1) if (x + dx, y + dy) in data_set))
        for x, y in x_xy + z_xy
    }
    layers = []
    for (xdx, xdy), (zdx, zdy) in zip(X_ORDER, Z_ORDER):
        layer = [(index(x, y), index(x + xdx, y + xdy)) for x, y in x_xy if (x + xdx, y + xdy) in data_set]
        layer += [(index(x + zdx, y + zdy), index(x, y)) for x, y in z_xy if (x + zdx, y + zdy) in data_set]
        layers.append(tuple(layer))

    return SurfaceGeometry(
        distance=d,
        coords=dict(sorted(coords.items())),
        data=tuple(sorted(index(*p) for p in data_xy)),
        ancillas=tuple(sorted(index(*p) for p in x_xy + z_xy)),
        x_ancillas=tuple(index(*p) for p in x_xy),
        z_ancillas=tuple(index(*p) for p in z_xy),
        support=support,
        cx_layers=tuple(layers),
        logical_z=tuple(index(x, 1) for x in range(1, n, 2)),
        logical_x=tuple(index(1, y) for y in range(1, n, 2)),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Noise
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoiseModel:
    """Circuit-level Pauli noise, with optional per-qubit and per-pair rates.

    The first four fields are the challenge's knobs (``challenge/surface_code.py:DEFAULT_NOISE``)
    and Stim's ``generated`` arguments, placed as the module docstring says.
    ``idle_during_measurement`` is the idle option and ``record_flip_probability`` the
    record-only readout flip, ``M(p)``; both are 0 unless set. ``idle_pauli_x``/``_y``/``_z``
    are the biased idle on the data, ``PAULI_CHANNEL_1``, and ``final_record_flip_probability``
    the record flip of the final readout layer (``None``: ``record_flip_probability``).
    ``idle_after_last_round`` is not a rate: True restores the idle after the last round's
    readout that circuits had before 2026-10-24, to rebuild an older decoder exactly.

    ``qubit_rates[q]`` maps any of those field names to a rate that replaces the default on
    qubit ``q``; ``after_clifford_depolarization`` there sets the ``DEPOLARIZE1`` after q's
    single-qubit gates. ``pair_rates[(a, b)]`` sets the ``DEPOLARIZE2`` after a CX on that pair,
    either way round; a pair without one uses ``after_clifford_depolarization``.
    """

    after_clifford_depolarization: float = 0.0
    after_reset_flip_probability: float = 0.0
    before_measure_flip_probability: float = 0.0
    before_round_data_depolarization: float = 0.0
    idle_during_measurement: float = 0.0
    record_flip_probability: float = 0.0
    qubit_rates: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    pair_rates: Mapping[tuple[int, int], float] = field(default_factory=dict)
    idle_pauli_x: float = 0.0
    idle_pauli_y: float = 0.0
    idle_pauli_z: float = 0.0
    final_record_flip_probability: float | None = None
    idle_after_last_round: bool = False

    def __post_init__(self):
        knobs = _knob_names()
        for name in knobs:
            if getattr(self, name) is not None:
                _check_rate(name, getattr(self, name))
        for q, rates in self.qubit_rates.items():
            for name, p in rates.items():
                if name not in knobs:
                    raise ValueError(f"qubit_rates[{q}]: unknown noise knob {name!r}; known: {', '.join(knobs)}")
                _check_rate(f"qubit_rates[{q}][{name!r}]", p)
        for q in (None, *self.qubit_rates):
            total = sum(self.idle_pauli(q)) if q is not None else self.idle_pauli_x + self.idle_pauli_y + self.idle_pauli_z
            if total > 1.0:
                raise ValueError(f"idle_pauli x + y + z = {total} > 1" + ("" if q is None else f" on qubit {q}"))
        pairs = {}
        for pair, p in self.pair_rates.items():
            a, b = pair
            _check_rate(f"pair_rates[{pair}]", p)
            pairs[frozenset((a, b))] = p
        object.__setattr__(self, "_pairs", pairs)

    @classmethod
    def uniform(cls, p: float, idle: float = 0.0) -> NoiseModel:
        """All four challenge knobs at ``p`` (``DEFAULT_NOISE`` is ``uniform(0.003)``)."""
        return cls(p, p, p, p, idle)

    @classmethod
    def from_dict(cls, rates: Mapping[str, float]) -> NoiseModel:
        """From a dict of knob names, such as the challenge's ``DEFAULT_NOISE``."""
        unknown = set(rates) - set(_knob_names()) - set(_NOT_KNOBS)
        if unknown:
            raise ValueError(f"unknown noise knobs {sorted(unknown)}; known: {', '.join(_knob_names())}")
        return cls(**rates)

    def rate(self, knob: str, qubit: int) -> float:
        return self.qubit_rates.get(qubit, {}).get(knob, getattr(self, knob))

    def pair_rate(self, a: int, b: int) -> float:
        return self._pairs.get(frozenset((a, b)), self.after_clifford_depolarization)

    def idle_pauli(self, qubit: int) -> tuple[float, float, float]:
        return tuple(self.rate(f"idle_pauli_{k}", qubit) for k in "xyz")  # type: ignore[return-value]

    def record_flip(self, qubit: int, *, final: bool = False) -> float:
        """The record flip of a readout of ``qubit``; ``final`` for the final readout layer."""
        if final:
            p = self.qubit_rates.get(qubit, {}).get("final_record_flip_probability", self.final_record_flip_probability)
            if p is not None:
                return p
        return self.rate("record_flip_probability", qubit)

    # Each method below appends one kind of channel at one place in the circuit. They are the
    # only code that writes noise, and a channel whose rate is 0 is left out.

    def after_reset(self, c: stim.Circuit, qubits: Iterable[int]) -> None:
        _append_per_qubit(c, "X_ERROR", qubits, lambda q: self.rate("after_reset_flip_probability", q))

    def before_measure(self, c: stim.Circuit, qubits: Iterable[int]) -> None:
        _append_per_qubit(c, "X_ERROR", qubits, lambda q: self.rate("before_measure_flip_probability", q))

    def after_single_qubit_gate(self, c: stim.Circuit, qubits: Iterable[int]) -> None:
        _append_per_qubit(c, "DEPOLARIZE1", qubits, lambda q: self.rate("after_clifford_depolarization", q))

    def after_cx(self, c: stim.Circuit, pairs: Iterable[tuple[int, int]]) -> None:
        groups: dict[float, list[int]] = {}
        for a, b in pairs:
            p = self.pair_rate(a, b)
            if p > 0:
                groups.setdefault(p, []).extend((a, b))
        for p, targets in groups.items():
            c.append("DEPOLARIZE2", targets, p)

    def round_start_data(self, c: stim.Circuit, data: Iterable[int]) -> None:
        _append_per_qubit(c, "DEPOLARIZE1", data, lambda q: self.rate("before_round_data_depolarization", q))

    def idle_data(self, c: stim.Circuit, data: Iterable[int]) -> None:
        data = list(data)
        _append_per_qubit(c, "DEPOLARIZE1", data, lambda q: self.rate("idle_during_measurement", q))
        groups: dict[tuple[float, float, float], list[int]] = {}
        for q in data:
            p = self.idle_pauli(q)
            if any(x > 0 for x in p):
                groups.setdefault(p, []).append(q)
        for p, targets in groups.items():
            c.append("PAULI_CHANNEL_1", targets, p)

    def measurement(self, c: stim.Circuit, gate: str, qubits: Iterable[int], *, final: bool = False) -> None:
        """``gate`` ("M" or "MR") on ``qubits``, in the order given, with the record flips.

        ``final`` marks the final readout layer, whose flips are ``final_record_flip_probability``.
        """
        qubits = list(qubits)
        rate = lambda q: self.record_flip(q, final=final)
        if gate == "MR":
            # the reset follows at once, so a record flip is a state flip just before MR
            _append_per_qubit(c, "X_ERROR", qubits, rate)
            c.append("MR", qubits)
            return
        for p, run in itertools.groupby(qubits, key=rate):
            if p > 0:
                c.append("M", list(run), p)
            else:
                c.append("M", list(run))

    def idle_live_ancillas(self, c: stim.Circuit, ancillas: Iterable[int]) -> None:
        _append_per_qubit(c, "X_ERROR", ancillas, lambda q: self.rate("idle_during_measurement", q))


#: NoiseModel fields that are not per-qubit rates.
_NOT_KNOBS = ("qubit_rates", "pair_rates", "idle_after_last_round")


def _knob_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(NoiseModel) if f.name not in _NOT_KNOBS)


def _check_rate(name: str, p: float) -> None:
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"{name} = {p} is not a probability")


def _append_per_qubit(c: stim.Circuit, gate: str, qubits: Iterable[int], rate_of) -> None:
    """One instruction per distinct nonzero rate, the qubits in the order given."""
    groups: dict[float, list[int]] = {}
    for q in qubits:
        p = rate_of(q)
        if p > 0:
            groups.setdefault(p, []).append(q)
    for p, targets in groups.items():
        c.append(gate, targets, p)


def _as_noise_model(noise) -> NoiseModel:
    if noise is None:
        return NoiseModel()
    if isinstance(noise, NoiseModel):
        return noise
    if isinstance(noise, Mapping):
        return NoiseModel.from_dict(noise)
    if isinstance(noise, (int, float)) and not isinstance(noise, bool):
        return NoiseModel.uniform(float(noise))
    raise TypeError(f"noise must be None, a NoiseModel, a dict of knob rates or a float, not {type(noise).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
#  The circuit
# ─────────────────────────────────────────────────────────────────────────────


def memory_circuit(
    distance: int,
    rounds: int,
    basis: str = "Z",
    no_reset: bool = True,
    noise=None,
) -> stim.Circuit:
    """A rotated surface-code memory experiment, as the M1 contract (section 1) specifies.

    Parameters
    ----------
    distance : odd code distance, at least 3 (d^2 data and d^2 - 1 measure qubits).
    rounds   : stabilizer rounds R, at least 1.
    basis    : "Z" prepares |0>^n and measures Z_L; "X" prepares |+>^n (``R`` then ``H``) and
               measures X_L (``H`` then ``M``).
    no_reset : True measures the ancillas with ``M`` and never resets them (Emerald); False uses
               ``MR``, for comparison in simulation.
    noise    : None for the hardware circuit (no noise instructions); else a ``NoiseModel``, a
               dict of its knobs (``challenge/surface_code.py:DEFAULT_NOISE`` works), or a float
               p for ``NoiseModel.uniform(p)``.

    Measurement order: each round's ancillas in ascending qubit index, then the data qubits in
    ascending qubit index. One observable, index 0. The detectors and their algebra are in the
    module docstring; their coordinates are (x, y, t), t = 0 for round 1 up to t = R for the
    final data detectors.
    """
    basis = _check_basis(basis)
    _check_rounds(rounds)
    g = geometry(distance)
    nm = _as_noise_model(noise)
    measure = "M" if no_reset else "MR"

    c = stim.Circuit()
    for q, (x, y) in g.coords.items():
        c.append("QUBIT_COORDS", [q], [x, y])

    # Hardware qubits start in |0>: this R is the only one.
    c.append("R", g.data)
    nm.after_reset(c, g.data)
    c.append("R", g.ancillas)
    nm.after_reset(c, g.ancillas)

    record: list[int] = []  # qubit measured at each measurement index
    last: dict[int, list[int]] = {a: [] for a in g.ancillas}  # ancilla -> its measurement indices

    def rec(indices: Iterable[int]) -> list[stim.GateTarget]:
        # most recent first, as Stim's generated circuits list them
        return [stim.target_rec(i - len(record)) for i in sorted(indices, reverse=True)]

    for k in range(1, rounds + 1):
        c.append("TICK")
        if k == 1 and basis == "X":
            c.append("H", g.data)  # |0> -> |+>, part of the preparation: no gate noise
        nm.round_start_data(c, g.data)
        c.append("H", sorted(g.x_ancillas))
        nm.after_single_qubit_gate(c, sorted(g.x_ancillas))
        c.append("TICK")
        for layer in g.cx_layers:
            c.append("CX", [q for pair in layer for q in pair])
            nm.after_cx(c, layer)
            c.append("TICK")
        c.append("H", sorted(g.x_ancillas))
        nm.after_single_qubit_gate(c, sorted(g.x_ancillas))
        if k == rounds and basis == "X":
            c.append("H", g.data)  # X_L readout, part of the measurement: no gate noise
        c.append("TICK")

        nm.before_measure(c, g.ancillas)
        nm.measurement(c, measure, g.ancillas, final=k == rounds)
        for a in g.ancillas:
            last[a].append(len(record))
            record.append(a)
        # the last round's ancillas are read in the data's layer: nothing waits after it
        idle = k < rounds or nm.idle_after_last_round
        if not no_reset:
            nm.after_reset(c, g.ancillas)
        elif idle:
            nm.idle_live_ancillas(c, g.ancillas)
        if idle:
            nm.idle_data(c, g.data)

        t = k - 1
        if k == 1:
            for a in g.basis_ancillas(basis):
                c.append("DETECTOR", rec([last[a][0]]), [*g.coords[a], t])
        else:
            for a in g.ancillas:
                m = last[a]  # m[k - 1] is m_k
                if no_reset:
                    refs = [m[k - 1]] + ([m[k - 3]] if k >= 3 else [])  # m_k + m_(k-2), m_0 = 0
                else:
                    refs = [m[k - 1], m[k - 2]]  # m_k + m_(k-1)
                c.append("DETECTOR", rec(refs), [*g.coords[a], t])

    nm.before_measure(c, g.data)
    nm.measurement(c, "M", g.data, final=True)
    data_at = {}
    for q in g.data:
        data_at[q] = len(record)
        record.append(q)

    for a in g.basis_ancillas(basis):
        m = last[a]
        final = [data_at[q] for q in g.support[a]] + [m[rounds - 1]]
        if no_reset and rounds >= 2:
            final.append(m[rounds - 2])
        c.append("DETECTOR", rec(final), [*g.coords[a], rounds])
    c.append("OBSERVABLE_INCLUDE", rec(data_at[q] for q in g.logical(basis)), 0)
    return c


def _check_distance(distance: int) -> None:
    if not isinstance(distance, int) or isinstance(distance, bool) or distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, not {distance!r}")


def _check_rounds(rounds: int) -> None:
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError(f"rounds must be an integer >= 1, not {rounds!r}")


def _check_basis(basis: str) -> str:
    b = str(basis).upper()
    if b not in ("Z", "X"):
        raise ValueError(f"basis must be 'Z' or 'X', not {basis!r}")
    return b
