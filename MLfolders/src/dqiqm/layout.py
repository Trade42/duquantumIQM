"""Stim surface-code circuits on fixed physical qubits of a square-grid IQM device, with zero SWAPs.

The architecture
    :class:`Architecture` is a device's qubit names (index i is the backend's physical qubit i)
    and its couplers, from a backend (:meth:`Architecture.from_backend`), from a
    :func:`dqiqm.hw.architecture` snapshot (:meth:`Architecture.from_dqa`), or given by hand. Its
    grid coordinates are inferred from the couplers alone (:func:`grid_coordinates`), never from
    qubit-name arithmetic.

The patch map
    Stim's rotated surface code puts data qubits at odd (x, y) and measure qubits at even (x, y),
    and every CX joins diagonal neighbours. ``(row, col) = ((y - x) / 2, (x + y) / 2)`` turns
    those diagonals into unit grid steps. A :class:`Placement` turns the patch by one of the eight
    symmetries of the square (``orientation``) and puts the top-left corner of its bounding box
    at ``origin``, a grid (row, col). :func:`placements` enumerates every valid one.

The converter
    :func:`native_circuit` turns a contract circuit (``_m1-contract.md``, section 1) into IQM's
    native gates on dense logical qubits, Stim measurement j in clbit j. :func:`to_hardware` puts
    it on a placement's physical qubits with ``transpile(..., initial_layout=...)`` and proves the
    result from the instructions IQM's serializer would send: no SWAP, one ``cz`` per CX, each on
    a coupler, and measurement j in clbit j on its qubit (:func:`check_hardware`).

Checks
    :func:`native_to_stim` reads a native circuit back into Stim with the source's detectors and
    observables, so that equivalence is checked exactly (:func:`check_equivalent`), not only by
    sampling. :func:`standin_circuit` is Stim's generated circuit in the contract's instruction
    set, the stand-in until ``dqiqm.surface`` lands.
"""

from __future__ import annotations

import cmath
import functools
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import stim
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RGate

Coord = tuple[int, int]

SEED = 2026
TARGETS = ("hardware", "simulation")

# The contract's instructions (section 1). Annotations and noise have no hardware equivalent.
_ANNOTATIONS = frozenset({"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"})
_NOISE = frozenset({"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"})

_I = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
# Simulation target only: Pauli gates pass through, so a check can inject a fault into the source.
_PAULIS = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_TOL = 1e-9


class PlacementError(ValueError):
    """The patch does not fit the architecture: a missing qubit, or a CX pair with no coupler."""


# ---------------------------------------------------------------------------
# The architecture and its grid.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Architecture:
    """A device's qubits, in the backend's index order, and its couplers, by name.

    ``fixed_coords``, when given, is the grid frame to use instead of inferring one: a device
    whose calibration turned couplers off keeps its full grid's frame, and so its placements'
    origins (:meth:`with_couplers`).
    """

    qubits: tuple[str, ...]
    couplers: frozenset[frozenset[str]]
    fixed_coords: tuple[tuple[str, Coord], ...] | None = None

    def __post_init__(self) -> None:
        known = set(self.qubits)
        if len(known) != len(self.qubits):
            raise ValueError("duplicate qubit names")
        for c in self.couplers:
            if len(c) != 2 or not c <= known:
                raise ValueError(f"coupler {sorted(c)} is not a pair of the architecture's qubits")

    @classmethod
    def from_pairs(cls, qubits: Iterable[str], couplers: Iterable[Sequence[str]]) -> Architecture:
        return cls(tuple(qubits), frozenset(frozenset(c) for c in couplers))

    @classmethod
    def from_backend(cls, backend: Any) -> Architecture:
        """An IQM qiskit backend's qubits (``index_to_qubit_name``) and its ``cz`` loci."""
        qubits = [backend.index_to_qubit_name(i) for i in range(backend.num_qubits)]
        return cls.from_pairs(qubits, backend.architecture.gates["cz"].loci)

    @classmethod
    def from_dqa(cls, dqa: Mapping[str, Any]) -> Architecture:
        """A dynamic quantum architecture as :func:`dqiqm.hw.architecture` returns it (``model_dump``)."""
        impls = dqa["gates"]["cz"]["implementations"].values()
        return cls.from_pairs(dqa["qubits"], (locus for impl in impls for locus in impl["loci"]))

    def with_couplers(self, couplers: Iterable[Sequence[str]]) -> Architecture:
        """The same qubits and grid frame with other couplers (say, the ones a calibration left on)."""
        return Architecture(self.qubits, frozenset(frozenset(c) for c in couplers), tuple(sorted(self.coords.items())))

    def coupled(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.couplers

    @functools.cached_property
    def coords(self) -> dict[str, Coord]:
        """Each qubit's grid (row, col): ``fixed_coords``, or inferred from the couplers (:func:`grid_coordinates`)."""
        if self.fixed_coords is None:
            return grid_coordinates(self)
        coords = dict(self.fixed_coords)
        _check_grid(self, coords)
        return coords

    def grid(self, mark: Mapping[str, str] | None = None) -> str:
        """The grid as text, one cell per site: the qubit's name, or ``mark[name]`` if given."""
        at = {rc: q for q, rc in self.coords.items()}
        rows = [r for r, _ in at]
        cols = [c for _, c in at]
        width = max(len(s) for s in [*self.qubits, *(mark or {}).values()])
        lines = []
        for r in range(min(rows), max(rows) + 1):
            cells = []
            for c in range(min(cols), max(cols) + 1):
                q = at.get((r, c))
                cells.append("." if q is None else (mark or {}).get(q, q))
            lines.append(" ".join(f"{s:>{width}}" for s in cells))
        return "\n".join(lines)


def _transform(rc: Coord, orientation: int) -> Coord:
    """One of the eight symmetries of the square: ``orientation % 4`` quarter turns, then a mirror if >= 4."""
    r, c = rc
    for _ in range(orientation % 4):
        r, c = c, -r
    return (r, -c) if orientation >= 4 else (r, c)


def _normalise(coords: Mapping[Any, Coord]) -> dict[Any, Coord]:
    r0 = min(r for r, _ in coords.values())
    c0 = min(c for _, c in coords.values())
    return {k: (r - r0, c - c0) for k, (r, c) in coords.items()}


def grid_coordinates(arch: Architecture) -> dict[str, Coord]:
    """Embed the coupler graph in the square grid: every coupler becomes a unit step.

    The unit cells are the graph's 4-cycles. The first cell is placed at the origin, and each
    cell that shares a coupler with a placed one goes on that coupler's other side. That fixes
    the embedding up to the square's symmetries; of those, the one that best keeps the qubit
    order row-major (row 0 at the top) is chosen, so the result is deterministic.

    A qubit in no 4-cycle gets no coordinate: no surface-code qubit can sit there, since every
    code qubit lies on a 4-cycle of CX pairs. Separate groups of cells are laid side by side,
    two columns apart. Raises ``ValueError`` if the couplers are not a subgraph of a square grid.
    """
    adj: dict[str, set[str]] = {q: set() for q in arch.qubits}
    for c in arch.couplers:
        a, b = sorted(c)
        adj[a].add(b)
        adj[b].add(a)
    order = {q: i for i, q in enumerate(arch.qubits)}

    # Each cell is (a, b, c, d), a 4-cycle a-b-c-d-a. edge_cells: coupler -> the cells it bounds.
    cells: dict[frozenset[str], tuple[str, str, str, str]] = {}
    for a in arch.qubits:
        for b, d in itertools.combinations(sorted(adj[a], key=order.get), 2):
            if d in adj[b]:
                raise ValueError(f"couplers {a}-{b}-{d} form a triangle: not a square grid")
            for c in sorted(adj[b] & adj[d] - {a}, key=order.get):
                if c in adj[a]:
                    raise ValueError(f"{a} and {c} are coupled across a cell: not a square grid")
                cells.setdefault(frozenset((a, b, c, d)), (a, b, c, d))
    edge_cells: dict[frozenset[str], list[tuple[str, str, str, str]]] = {}
    for cyc in cells.values():
        for i in range(4):
            edge_cells.setdefault(frozenset((cyc[i], cyc[(i + 1) % 4])), []).append(cyc)
    for e, cs in edge_cells.items():
        if len(cs) > 2:
            raise ValueError(f"coupler {sorted(e)} bounds {len(cs)} cells: not a square grid")

    def neighbour_in(cyc: tuple[str, ...], q: str, other: str) -> str:
        i = cyc.index(q)
        left, right = cyc[(i - 1) % 4], cyc[(i + 1) % 4]
        return right if left == other else left

    components: list[dict[str, Coord]] = []
    done: set[frozenset[str]] = set()
    for key in sorted(cells, key=lambda k: sorted(order[q] for q in k)):
        if key in done:
            continue
        a, b, c, d = cells[key]
        pos: dict[str, Coord] = {a: (0, 0), b: (0, 1), c: (1, 1), d: (1, 0)}
        done.add(key)
        queue = [cells[key]]
        while queue:
            cyc = queue.pop()
            for i in range(4):
                p, q = cyc[i], cyc[(i + 1) % 4]
                for cyc2 in edge_cells[frozenset((p, q))]:
                    key2 = frozenset(cyc2)
                    if key2 in done:
                        continue
                    # cyc2 lies on the other side of the shared coupler p-q
                    p1, q1 = neighbour_in(cyc, p, q), neighbour_in(cyc, q, p)
                    w = (pos[p1][0] - pos[p][0], pos[p1][1] - pos[p][1])
                    p2, q2 = neighbour_in(cyc2, p, q), neighbour_in(cyc2, q, p)
                    for s, base in ((p2, p), (q2, q)):
                        new = (pos[base][0] - w[0], pos[base][1] - w[1])
                        if pos.setdefault(s, new) != new:
                            raise ValueError(f"{s} lands on two grid sites: not a square grid")
                    done.add(key2)
                    queue.append(cyc2)
        components.append(pos)

    def concordance(pos: Mapping[str, Coord]) -> int:
        qs = sorted(pos, key=order.get)
        return sum(pos[x] < pos[y] for x, y in itertools.combinations(qs, 2))

    coords: dict[str, Coord] = {}
    offset = 0
    for pos in sorted(components, key=lambda p: min(order[q] for q in p)):
        candidates = [_normalise({q: _transform(rc, k) for q, rc in pos.items()}) for k in range(8)]
        best = max(candidates, key=concordance)  # max keeps the first on ties: deterministic
        coords.update({q: (r, c + offset) for q, (r, c) in best.items()})
        offset = max(c for _, c in coords.values()) + 2

    _check_grid(arch, coords)
    return coords


def _check_grid(arch: Architecture, coords: Mapping[str, Coord]) -> None:
    """Raise ``ValueError`` unless the sites are distinct and every coupler between them is a unit step."""
    sites: dict[Coord, str] = {}
    for q, rc in coords.items():
        if q not in arch.qubits:
            raise ValueError(f"{q} has a grid site but is not one of the architecture's qubits")
        if rc in sites:
            raise ValueError(f"{sites[rc]} and {q} land on the same grid site: not a square grid")
        sites[rc] = q
    for c in arch.couplers:
        a, b = sorted(c)
        if a in coords and b in coords:
            (ra, ca), (rb, cb) = coords[a], coords[b]
            if abs(ra - rb) + abs(ca - cb) != 1:
                raise ValueError(f"coupler {a}-{b} is not a unit grid step")


# ---------------------------------------------------------------------------
# The code's grid and its placements.
# ---------------------------------------------------------------------------


def _gate_qubits(circuit: stim.Circuit) -> list[int]:
    """The qubits the circuit's gates act on (not annotations or noise), ascending."""
    used: set[int] = set()
    for inst in circuit.flattened():
        if inst.name in _ANNOTATIONS or inst.name in _NOISE or inst.name == "TICK":
            continue
        used.update(t.value for t in inst.targets_copy() if t.is_qubit_target)
    return sorted(used)


def cx_pairs(circuit: stim.Circuit) -> list[tuple[int, int]]:
    """Every CX as (control, target), in circuit order, repeats included."""
    pairs = []
    for inst in circuit.flattened():
        if inst.name == "CX":
            ts = [t.value for t in inst.targets_copy()]
            pairs.extend(zip(ts[::2], ts[1::2]))
    return pairs


def measured_qubits(circuit: stim.Circuit) -> list[int]:
    """The qubit of each Stim measurement, in record order: entry j is measurement j."""
    out = []
    for inst in circuit.flattened():
        if inst.name in ("M", "MR"):
            out.extend(t.value for t in inst.targets_copy())
    return out


def code_coordinates(circuit: stim.Circuit) -> dict[int, Coord]:
    """Each qubit of a rotated surface-code circuit at its grid (row, col), the bounding box at (0, 0).

    ``(row, col) = ((y - x) / 2, (x + y) / 2)`` from the ``QUBIT_COORDS``, which must be in
    Stim's convention for the rotated code (data at odd x and y, measure qubits at even x and y).
    Raises ``ValueError`` if a CX pair does not become a unit grid step.
    """
    xy = circuit.get_final_qubit_coordinates()
    rc: dict[int, Coord] = {}
    for q in _gate_qubits(circuit):
        if q not in xy or len(xy[q]) < 2:
            raise ValueError(f"qubit {q} has no QUBIT_COORDS")
        x, y = xy[q][0], xy[q][1]
        if not (float(x).is_integer() and float(y).is_integer()) or (int(x) + int(y)) % 2:
            raise ValueError(f"qubit {q} at {(x, y)} is off the rotated code's lattice (x + y must be even)")
        rc[q] = ((int(y) - int(x)) // 2, (int(x) + int(y)) // 2)
    rc = _normalise(rc)
    for a, b in cx_pairs(circuit):
        if abs(rc[a][0] - rc[b][0]) + abs(rc[a][1] - rc[b][1]) != 1:
            raise ValueError(f"CX {a} {b} joins qubits that are not diagonal neighbours")
    return rc


@dataclass(frozen=True)
class Placement:
    """The patch turned by ``orientation`` (0-7), its bounding box's top-left corner at grid ``origin``."""

    origin: Coord
    orientation: int
    mapping: tuple[tuple[int, str], ...]  # (Stim qubit, device qubit name), by Stim qubit

    @property
    def physical(self) -> dict[int, str]:
        """Stim qubit index -> device qubit name."""
        return dict(self.mapping)

    @property
    def qubit_set(self) -> frozenset[str]:
        return frozenset(name for _, name in self.mapping)


def _try_place(
    code: Mapping[int, Coord],
    pairs: Iterable[tuple[int, int]],
    arch: Architecture,
    sites: Mapping[Coord, str],
    origin: Coord,
    orientation: int,
) -> tuple[Placement | None, str]:
    patch = _normalise({q: _transform(rc, orientation) for q, rc in code.items()})
    mapping = {}
    for q, (r, c) in patch.items():
        name = sites.get((origin[0] + r, origin[1] + c))
        if name is None:
            return None, f"Stim qubit {q} lands on grid site {(origin[0] + r, origin[1] + c)}, which has no qubit"
        mapping[q] = name
    for a, b in pairs:
        if not arch.coupled(mapping[a], mapping[b]):
            return None, f"CX {a} {b} lands on {mapping[a]}-{mapping[b]}, which is not a coupler"
    return Placement(origin, orientation, tuple(sorted(mapping.items()))), ""


def place(circuit: stim.Circuit, arch: Architecture, origin: Coord, orientation: int) -> Placement:
    """The placement at ``origin`` and ``orientation``; raises :class:`PlacementError` if it does not fit."""
    if orientation not in range(8):
        raise ValueError(f"orientation must be 0-7, got {orientation!r}")
    sites = {rc: q for q, rc in arch.coords.items()}
    found, why = _try_place(code_coordinates(circuit), set(cx_pairs(circuit)), arch, sites, tuple(origin), orientation)
    if found is None:
        raise PlacementError(why)
    return found


def placements(circuit: stim.Circuit, arch: Architecture) -> list[Placement]:
    """Every placement of the circuit's patch on the architecture with every CX pair on a coupler.

    Sorted by (orientation, origin). Placements related by a symmetry of the patch use the same
    qubits with the X and Z checks swapped or moved, so they are listed separately.
    """
    code = code_coordinates(circuit)
    pairs = set(cx_pairs(circuit))
    sites = {rc: q for q, rc in arch.coords.items()}
    anchor = min(code)
    found = []
    for orientation in range(8):
        patch = _normalise({q: _transform(rc, orientation) for q, rc in code.items()})
        ar, ac = patch[anchor]
        for r, c in sorted({(r - ar, c - ac) for r, c in sites}):
            p, _ = _try_place(code, pairs, arch, sites, (r, c), orientation)
            if p is not None:
                found.append(p)
    return found


def placement_grid(placement: Placement, arch: Architecture, circuit: stim.Circuit) -> str:
    """The architecture's grid with the patch marked: D<q> for data, X<q>/Z<q> for measure qubits (Stim index q)."""
    kinds = _check_kinds(circuit)
    mark = {name: f"{kinds.get(q, 'D')}{q}" for q, name in placement.mapping}
    return arch.grid(mark)


def _check_kinds(circuit: stim.Circuit) -> dict[int, str]:
    """Measure qubit -> 'X' if it is a CX control (an X check), 'Z' if a CX target. Data qubits are absent."""
    xy = circuit.get_final_qubit_coordinates()
    measure = {q for q in _gate_qubits(circuit) if q in xy and int(xy[q][0]) % 2 == 0}
    kinds = {}
    for c, t in cx_pairs(circuit):
        if c in measure:
            kinds[c] = "X"
        if t in measure:
            kinds[t] = "Z"
    return kinds


# ---------------------------------------------------------------------------
# Stim -> IQM native gates.
# ---------------------------------------------------------------------------


def _rz(alpha: float) -> np.ndarray:
    return np.diag([cmath.exp(-0.5j * alpha), cmath.exp(0.5j * alpha)])


def _prx(theta: float, phi: float) -> np.ndarray:
    """IQM's PRX, qiskit's ``RGate(theta, phi)``: exp(-i theta/2 (cos(phi) X + sin(phi) Y))."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * cmath.exp(-1j * phi) * s], [-1j * cmath.exp(1j * phi) * s, c]])


def _snap(angle: float) -> float:
    """The angle in (-pi, pi], snapped to a multiple of pi/2 when it is one to within 1e-9."""
    a = math.remainder(angle, 2 * math.pi)
    k = round(a / (math.pi / 2))
    if abs(a - k * math.pi / 2) < _TOL:
        a = k * math.pi / 2
    return math.pi if a <= -math.pi + _TOL else a


def _same_up_to_phase(u: np.ndarray, v: np.ndarray) -> bool:
    return abs(abs(np.trace(u.conj().T @ v)) - 2) < 1e-8


def prx_then_rz(u: np.ndarray) -> tuple[float, float, float]:
    """(theta, phi, alpha) with ``u`` = RZ(alpha) . PRX(theta, phi) up to global phase, theta in [0, pi].

    In time order: PRX(theta, phi), then RZ(alpha). theta is 0 exactly when ``u`` is diagonal.
    """
    u = np.asarray(u, dtype=complex)
    u = u / cmath.sqrt(np.linalg.det(u))
    c, s = abs(u[0, 0]), abs(u[1, 0])
    if s < _TOL:
        theta, phi, alpha = 0.0, 0.0, cmath.phase(u[1, 1]) - cmath.phase(u[0, 0])
    elif c < _TOL:
        theta, alpha = math.pi, 0.0
        phi = cmath.phase(u[1, 0] / u[0, 1]) / 2
    else:
        theta = 2 * math.atan2(s, c)
        alpha = cmath.phase(u[1, 1]) - cmath.phase(u[0, 0])
        v00 = u[0, 0] * cmath.exp(0.5j * alpha)
        v10 = u[1, 0] * cmath.exp(-0.5j * alpha)
        phi = cmath.phase(1j * v10 / v00)
    theta, phi, alpha = abs(_snap(theta)), _snap(phi), _snap(alpha)
    if not _same_up_to_phase(u, _rz(alpha) @ _prx(theta, phi)):
        raise AssertionError(f"PRX/RZ decomposition failed for {u!r}")
    return theta, phi, alpha


def native_circuit(
    circuit: stim.Circuit, *, target: str = "hardware", barriers: bool = True, name: str | None = None
) -> tuple[QuantumCircuit, tuple[int, ...]]:
    """A contract circuit in IQM's native gates: ``r`` (PRX), ``cz``, ``measure``, ``barrier``.

    Returns the circuit on dense logical qubits and the Stim qubit of each: logical qubit i is
    Stim qubit ``stim_qubits[i]`` (ascending). Stim measurement j is written to clbit j.

    Each CX(c, t) is H(t) CZ(c, t) H(t). The single-qubit gates are not emitted as they come:
    each qubit carries its pending gate U, a 2x2 unitary, and only when a CZ or a measurement
    reaches the qubit is U written out, as U = RZ(alpha) PRX(theta, phi): the PRX is emitted (none
    when U is diagonal) and RZ(alpha) stays pending. That is exact: RZ commutes with CZ, it
    merges into the next pending gate, and before a Z measurement it changes nothing, so it is
    dropped there (a virtual Z). So every run of single-qubit gates between two CZs or
    measurements of a qubit costs one ``r`` gate, or none when it is diagonal (H H between two CX
    on the same target cancels). With one CZ per CX in the source's order, that is the fewest
    possible: a run that is not diagonal cannot be done with zero PRX.

    With ``barriers=True`` (the default) each ``TICK`` becomes a barrier over all the circuit's
    qubits, unless nothing was emitted since the last one: the source's layers stay layers and
    each CZ layer runs in lockstep. A readout layer gets one more barrier, between its rotations
    and its measurements, so each round's measurements are one parallel readout. With
    ``barriers=False`` only each qubit's order of operations is kept, which saves two layers of
    depth per round but spreads a round's measurements over several layers.

    Each ``r`` goes as early as it can, at the start of the first layer after the qubit's
    previous CZ or measurement: when the qubit idles in that layer, the ``r`` runs alongside the
    other qubits' CZs and adds no depth. The exception is the ``r`` right before a measurement,
    which goes as late as it can, so the readout stays aligned. ``R`` is dropped when it is the
    qubit's first operation, since hardware qubits start in |0>.

    ``target="hardware"`` raises on ``MR``, on a later ``R`` (Emerald has no mid-circuit reset),
    on every noise instruction, ``M(p)`` with p > 0 included (contract, amended 18:15 UTC), and
    on anything outside the contract. ``target="simulation"`` turns ``MR`` into measure then reset
    and a later ``R`` into reset, strips the noise (``M(p)`` becomes a plain measure: the result is
    the noiseless circuit), and lets X, Y and Z through, for Aer only.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    flat = circuit.flattened()
    stim_qubits = tuple(_gate_qubits(flat))
    dense = {q: i for i, q in enumerate(stim_qubits)}
    pending = [_I.copy() for _ in stim_qubits]
    touched = [False] * len(stim_qubits)
    clbit = 0
    # Layers between barriers. Each is (early r gates on qubits idle until then, then the rest in
    # order). last[i] is the layer of qubit i's latest CZ, measurement or reset (-1: none yet).
    layers: list[tuple[list, list]] = [([], [])]
    last = [-1] * len(stim_qubits)

    def emit(item: tuple, qubits: Sequence[int]) -> None:
        layers[-1][1].append(item)
        for i in qubits:
            last[i] = len(layers) - 1

    def flush(i: int, *, late: bool = False) -> None:
        theta, phi, alpha = prx_then_rz(pending[i])
        if theta:
            gate = ("r", (theta, phi), (i,), ())
            if last[i] == len(layers) - 1:
                layers[-1][1].append(gate)  # after the qubit's own earlier operation in this layer
            elif late:
                layers[-1][0].append(gate)  # right before a measurement: the readout stays aligned
            else:
                layers[last[i] + 1][0].append(gate)  # the qubit idles from there until now
        pending[i] = _rz(alpha)

    for inst in flat:
        op = inst.name
        noisy = op in _NOISE or (op in ("M", "MR") and any(p > 0 for p in inst.gate_args_copy()))
        if noisy and target == "hardware":
            raise ValueError(f"{inst}: noise has no place in a hardware circuit (build it with noise=None)")
        if op in _ANNOTATIONS or op in _NOISE:
            continue
        if op == "TICK":
            if layers[-1] != ([], []):
                layers.append(([], []))
            continue
        targets = inst.targets_copy()
        if any(not t.is_qubit_target or t.is_inverted_result_target for t in targets):
            raise ValueError(f"{inst}: only plain qubit targets are supported")
        qs = [dense[t.value] for t in targets]
        if op == "R":
            for i in qs:
                if touched[i]:
                    if target == "hardware":
                        raise ValueError(f"R on qubit {stim_qubits[i]} after the start: Emerald has no mid-circuit reset")
                    emit(("reset", (), (i,), ()), [i])
                pending[i] = _I.copy()
        elif op == "H" or (op in _PAULIS and target == "simulation"):
            gate = _H if op == "H" else _PAULIS[op]
            for i in qs:
                pending[i] = gate @ pending[i]
                touched[i] = True
        elif op == "CX":
            for c, t in zip(qs[::2], qs[1::2]):
                pending[t] = _H @ pending[t]
                flush(c)
                flush(t)
                emit(("cz", (), (c, t), ()), [c, t])
                pending[t] = _H @ pending[t]
                touched[c] = touched[t] = True
        elif op in ("M", "MR"):
            if op == "MR" and target == "hardware":
                raise ValueError("MR: Emerald has no mid-circuit reset (use target='simulation' for Aer)")
            for i in qs:
                flush(i, late=True)
                pending[i] = _I.copy()  # the virtual Z before a Z measurement
                emit(("measure", (), (i,), (clbit,)), [i])
                clbit += 1
                if op == "MR":
                    emit(("reset", (), (i,), ()), [i])
                touched[i] = True
        else:
            raise ValueError(f"{op} is outside the contract's instructions (target={target!r})")
    for i, u in enumerate(pending):
        if not _same_up_to_phase(u, np.diag(np.diag(u))):
            raise ValueError(f"qubit {stim_qubits[i]} ends with a gate that is not followed by a measurement")
    assert clbit == flat.num_measurements

    qc = QuantumCircuit(len(stim_qubits), flat.num_measurements, name=name)
    filled = [layer for layer in layers if layer != ([], [])]
    for n, (early, rest) in enumerate(filled):
        # Without the barrier the layers merge, and each qubit keeps its order: a layer's early r
        # gates are on qubits whose previous operation is in an earlier layer.
        if n and barriers:
            qc.barrier()
        items = [*early, *rest]
        kinds = [item[0] for item in rest]
        if barriers and "measure" in kinds and "r" not in kinds and "cz" not in kinds:
            # a readout layer: its rotations, a barrier, then every measurement at once
            for _, params, qubits, _ in early:
                qc.append(RGate(*params), list(qubits))
            if early:
                qc.barrier()
            items = rest
        for op, params, qubits, clbits in items:
            if op == "r":
                qc.append(RGate(*params), list(qubits))
            elif op == "cz":
                qc.cz(*qubits)
            elif op == "measure":
                qc.measure(qubits[0], clbits[0])
            else:
                qc.reset(qubits[0])
    return qc, stim_qubits


@dataclass(frozen=True)
class HardwareCircuit:
    """A contract circuit on a placement's physical qubits, ready for ``dqiqm.hw.submit``.

    ``hw.submit(hc.circuit, qubit_index_to_name=hc.qubit_index_to_name, ...)`` (hardware target only).
    """

    circuit: QuantumCircuit  # transpiled; the serializer places it through its layout
    qubit_index_to_name: dict[int, str]  # physical qubit index -> device qubit name, for the used qubits
    placement: Placement
    stim_qubits: tuple[int, ...]  # logical qubit i of ``logical`` is Stim qubit stim_qubits[i]
    logical: QuantumCircuit  # the native circuit before placement
    initial_layout: tuple[int, ...]  # logical qubit i -> physical qubit index
    target: str


def to_hardware(
    circuit: stim.Circuit,
    backend: Any,
    placement: Placement,
    *,
    target: str = "hardware",
    barriers: bool = True,
    name: str | None = None,
    seed_transpiler: int = SEED,
) -> HardwareCircuit:
    """Convert ``circuit`` (:func:`native_circuit`) and transpile it onto ``placement``'s qubits of ``backend``.

    The hardware target is transpiled against the backend itself at optimization level 0: the
    gates are already native and already minimal, so the transpiler only places them (IQM's
    level-0 scheduling stage rebuilds the circuit around its layout, which the serializer
    reads). The simulation target has resets, which the IQM target lacks, so it is transpiled
    against the backend's coupling map with ``reset`` added to its basis; it is for Aer only.
    The result is checked with :func:`check_hardware` before it is returned.
    """
    native, stim_qubits = native_circuit(circuit, target=target, barriers=barriers, name=name)
    physical = placement.physical
    missing = [q for q in stim_qubits if q not in physical]
    if missing:
        raise PlacementError(f"Stim qubits {missing} are not in the placement")
    initial = [backend.qubit_name_to_index(physical[q]) for q in stim_qubits]
    if target == "hardware":
        placed = transpile(
            native, backend=backend, initial_layout=initial, optimization_level=0, seed_transpiler=seed_transpiler
        )
    else:
        placed = transpile(
            native,
            coupling_map=backend.coupling_map,
            basis_gates=[*backend.operation_names, "reset"],
            initial_layout=initial,
            optimization_level=0,
            seed_transpiler=seed_transpiler,
        )
    hc = HardwareCircuit(
        circuit=placed,
        qubit_index_to_name={p: backend.index_to_qubit_name(p) for p in initial},
        placement=placement,
        stim_qubits=stim_qubits,
        logical=native,
        initial_layout=tuple(initial),
        target=target,
    )
    check_hardware(hc, circuit, Architecture.from_backend(backend))
    return hc


# ---------------------------------------------------------------------------
# Checks.
# ---------------------------------------------------------------------------


def sent_instructions(circuit: QuantumCircuit, qubit_index_to_name: Mapping[int, str]) -> list[Any]:
    """The instructions IQM's serializer makes of ``circuit``: what the server would receive. Offline."""
    from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions

    return serialize_instructions(circuit, dict(qubit_index_to_name))


def _per_qubit(instructions: Iterable[Any]) -> dict[str, list[tuple[str, tuple[str, ...], tuple]]]:
    """Each qubit's sequence of (op, locus, args): equal sequences on every qubit mean the same circuit."""
    seqs: dict[str, list] = {}
    for inst in instructions:
        args = tuple(sorted((k, round(v, 12) if isinstance(v, float) else v) for k, v in inst.args.items()))
        # cz and barrier are symmetric, and IQM's level-0 rebuild may reorder their loci
        locus = tuple(sorted(inst.locus)) if inst.name in ("cz", "barrier") else tuple(inst.locus)
        for q in inst.locus:
            seqs.setdefault(q, []).append((inst.name, locus, args))
    return seqs


def check_hardware(hc: HardwareCircuit, source: stim.Circuit, arch: Architecture) -> dict[str, int]:
    """Prove a placed circuit from the instructions the serializer would send, or raise ``AssertionError``.

    - the layout is the one asked for (the transpiler did not move the patch);
    - no ``swap``, and exactly one ``cz`` per CX of the source;
    - every ``cz`` locus is a coupler of ``arch``;
    - only native operations (``prx``, ``cz``, ``measure``, ``barrier``; ``reset`` on the simulation target);
    - Stim measurement j is written to clbit j and measures the device qubit its Stim qubit is placed on;
    - every qubit's instruction sequence equals the logical circuit's, mapped through the placement.

    Returns the operation counts of the sent instructions.
    """
    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey

    layout = hc.circuit.layout
    assert layout is not None, "the transpiled circuit has no layout"
    got = list(layout.initial_index_layout(filter_ancillas=True))
    assert got == list(hc.initial_layout), f"the transpiler moved the patch: {got} != {list(hc.initial_layout)}"
    assert "swap" not in hc.circuit.count_ops(), "a SWAP was inserted"

    sent = sent_instructions(hc.circuit, hc.qubit_index_to_name)
    allowed = {"prx", "cz", "measure", "barrier"} | ({"reset"} if hc.target == "simulation" else set())
    counts: dict[str, int] = {}
    for inst in sent:
        assert inst.name in allowed, f"{inst.name} is not a native operation for the {hc.target} target"
        counts[inst.name] = counts.get(inst.name, 0) + 1
        if inst.name == "cz":
            assert arch.coupled(*inst.locus), f"cz on {inst.locus} is not on a coupler"
    n_cx = len(cx_pairs(source))
    assert counts.get("cz", 0) == n_cx, f"{counts.get('cz', 0)} cz for {n_cx} CX"

    physical = hc.placement.physical
    expect = {j: physical[q] for j, q in enumerate(measured_qubits(source))}
    measured: dict[int, str] = {}
    for inst in sent:
        if inst.name == "measure":
            j = MeasurementKey.from_string(inst.args["key"]).clbit_idx
            assert j not in measured, f"clbit {j} is written twice"
            measured[j] = inst.locus[0]
    assert measured == expect, "Stim measurement j does not land in clbit j on its placed qubit"

    logical_names = {i: physical[q] for i, q in enumerate(hc.stim_qubits)}
    assert _per_qubit(sent) == _per_qubit(sent_instructions(hc.logical, logical_names)), (
        "the transpiled circuit differs from the logical circuit"
    )
    return counts


@functools.cache
def _clifford_names() -> dict[str, str]:
    """str(stim.Tableau) -> the name of the single-qubit Clifford gate it is."""
    return {
        str(g.tableau): name
        for name, g in stim.gate_data().items()
        if g.is_unitary and g.is_single_qubit_gate and g.tableau is not None
    }


def native_to_stim(qc: QuantumCircuit, source: stim.Circuit) -> tuple[stim.Circuit, list[int]]:
    """A native circuit (Clifford ``r`` angles) back in Stim, with ``source``'s detectors and observable.

    Qubits keep their circuit indices; ``barrier`` becomes ``TICK``. Returns the Stim circuit and
    ``clbits``: Stim's measurement k writes clbit ``clbits[k]``. The source's detectors refer to
    its measurement j, which is clbit j of ``qc``, so they are rewritten through ``clbits``.
    A detector of the result is deterministic only if the native circuit is right, so
    ``detector_error_model()`` on the noiseless result checks every detector exactly.
    """
    names = _clifford_names()
    out = stim.Circuit()
    clbits: list[int] = []
    for ci in qc.data:
        op = ci.operation.name
        qs = [qc.find_bit(q).index for q in ci.qubits]
        if op == "r":
            tab = stim.Tableau.from_unitary_matrix(_prx(*map(float, ci.operation.params)), endian="little")
            if str(tab) not in names:
                raise ValueError(f"r{tuple(ci.operation.params)} is not a Clifford gate")
            out.append(names[str(tab)], qs)
        elif op == "cz":
            out.append("CZ", qs)
        elif op == "measure":
            out.append("M", qs)
            clbits.append(qc.find_bit(ci.clbits[0]).index)
        elif op == "reset":
            out.append("R", qs)
        elif op == "barrier":
            out.append("TICK")
        else:
            raise ValueError(f"{op} has no Stim equivalent here")
    if sorted(clbits) != list(range(source.num_measurements)):
        raise ValueError("the circuit does not write each of the source's measurements exactly once")
    k_of = {j: k for k, j in enumerate(clbits)}
    n = len(clbits)
    m = 0
    for inst in source.flattened():
        if inst.name in ("M", "MR"):
            m += len(inst.targets_copy())
        elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            recs = [stim.target_rec(k_of[m + t.value] - n) for t in inst.targets_copy()]
            out.append(inst.name, recs, inst.gate_args_copy() if inst.name == "OBSERVABLE_INCLUDE" else [])
    return out, clbits


def check_equivalent(qc: QuantumCircuit, source: stim.Circuit, *, shots: int = 64, seed: int = SEED) -> None:
    """Raise ``AssertionError`` unless the noiseless native circuit keeps every detector and observable of ``source``.

    Exact for determinism: ``detector_error_model()`` of the circuit read back into Stim raises
    if any detector or observable is not deterministic. Then its sampled records, put in clbit
    order and run through the source's own ``compile_m2d_converter``, must fire nothing, so each
    deterministic parity also has the source's value.
    """
    rt, clbits = native_to_stim(qc, source)
    try:
        rt.detector_error_model()
    except ValueError as e:
        raise AssertionError(f"a detector or observable is not deterministic: {e}") from None
    sample = rt.compile_sampler(seed=seed).sample(shots)
    record = np.zeros_like(sample)
    record[:, clbits] = sample
    det, obs = source.compile_m2d_converter().convert(measurements=record, separate_observables=True)
    assert not det.any(), f"{int(det.sum())} detection events in a noiseless run"
    assert not obs.any(), f"{int(obs.sum())} observable flips in a noiseless run"


def insert_pauli(circuit: stim.Circuit, tick: int, qubit: int, pauli: str) -> stim.Circuit:
    """``circuit`` (flattened) with ``pauli`` on ``qubit`` right after its ``tick``-th TICK (0-based)."""
    out = stim.Circuit()
    seen = 0
    for inst in circuit.flattened():
        out.append(inst)
        if inst.name == "TICK":
            if seen == tick:
                out.append(pauli, [qubit])
            seen += 1
    if seen <= tick:
        raise ValueError(f"the circuit has only {seen} TICKs")
    return out


# ---------------------------------------------------------------------------
# The stand-in source circuit.
# ---------------------------------------------------------------------------


def standin_circuit(distance: int, rounds: int, basis: str = "Z", *, reset: bool = True) -> stim.Circuit:
    """Stim's generated rotated memory circuit in the contract's instruction set (stand-in until ``dqiqm.surface``).

    ``RX q`` becomes ``R q`` then ``H q``, and ``MX q`` becomes ``H q`` then ``M q`` (contract,
    section 1). With ``reset=False`` every ``MR`` becomes ``M``: the gates and qubits of the
    hardware target, but its detectors are still the with-reset ones, which fire on about half
    the shots without reset. Use that form only for layout and gate counts, never for detectors.
    """
    basis = basis.upper()
    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be X or Z, got {basis!r}")
    gen = stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", distance=distance, rounds=rounds)
    out = stim.Circuit()
    for inst in gen.flattened():
        targets = inst.targets_copy()
        if inst.name == "RX":
            out.append("R", targets)
            out.append("H", targets)
        elif inst.name == "MX":
            out.append("H", targets)
            out.append("M", targets)
        elif inst.name == "MR" and not reset:
            out.append("M", targets)
        else:
            out.append(inst)
    return out
