"""Exhaustive placement over square-grid translations and eight symmetries."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math


def interaction_pairs(circuit):
    return [(a.value, b.value) for i in circuit.flattened() if i.name in {"CX", "CZ", "SWAP", "ISWAP"}
            for a, b in zip(i.targets_copy()[::2], i.targets_copy()[1::2])]


def rotated_positions(circuit):
    """Stim's diagonal data-check edges -> square-grid unit edges."""
    positions = {}
    for q, xy in circuit.get_final_qubit_coordinates().items():
        x, y = xy[:2]
        u, v = (x+y)/2, (x-y)/2
        if u != int(u) or v != int(v):
            raise ValueError("Expected rotated surface-code coordinates.")
        positions[q] = (int(u), int(v))
    return positions


def validate_layout(circuit, sparse_map, couplings):
    edges = {frozenset(e) for e in couplings}
    if set(rotated_positions(circuit)) - set(sparse_map):
        raise ValueError("Layout is missing circuit qubits.")
    if len(set(sparse_map.values())) != len(sparse_map):
        raise ValueError("Layout must be injective.")
    bad = [(a, b) for a, b in interaction_pairs(circuit)
           if frozenset((sparse_map[a], sparse_map[b])) not in edges]
    if bad:
        raise ValueError(f"Non-native interaction(s), refusing SWAP routing: {bad[:5]}")
    return dict(two_qubit_gates=len(interaction_pairs(circuit)), non_native_pairs=0,
                routing_swaps=0, physical_qubits=len(sparse_map))


@dataclass(frozen=True)
class Placement:
    stim_to_physical: dict[int, str]
    score: float
    orientation: int
    origin: tuple[int, int]

    def dense_layout(self, physical_order):
        indices = {q: i for i, q in enumerate(physical_order)}
        return [indices[self.stim_to_physical[q]] for q in sorted(self.stim_to_physical)]


def rank_placements(circuit, *, calibration=None, rows=8, cols=8, grid=None,
                    couplings=None, readout_weight=1.0, coherence_weight=1.0,
                    leakage_weight=0.0):
    """Rank all legal geometric placements; optimum only within this search set.

    A supplied grid holds QB names or None for missing locations. Actual backend
    couplings are authoritative. No chip-numbering assumption bypasses validation.
    """
    if min(readout_weight, coherence_weight, leakage_weight) < 0:
        raise ValueError("Placement weights must be nonnegative.")
    if grid is None:
        grid = [[f"QB{r*cols+c+1}" for c in range(cols)] for r in range(rows)]
    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("Grid must be rectangular.")
    if couplings is None:
        couplings = ([key.split("|") for key in calibration.edges] if calibration else
            [(grid[r][c], grid[rr][cc]) for r in range(rows) for c in range(cols)
             for rr, cc in ((r+1, c), (r, c+1)) if rr < rows and cc < cols
             and grid[r][c] is not None and grid[rr][cc] is not None])
    positions = rotated_positions(circuit)
    one, measure, reset, idle_ns = Counter(), Counter(), Counter(), Counter()
    pair_counts = Counter(tuple(sorted(pair)) for pair in interaction_pairs(circuit))
    moment, duration = set(), 0
    def finish_moment():
        for q in positions:
            if q not in moment:
                idle_ns[q] += duration
        moment.clear()
    for inst in circuit.flattened():
        name = inst.name
        if name == "TICK":
            finish_moment()
            duration = 0
            continue
        if name in {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}:
            continue
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        moment.update(qs)
        if name in {"M", "MX", "MR", "MRX"}:
            measure.update(qs)
            if name.startswith("MR"):
                reset.update(qs)
            key = "measure"
        elif name in {"R", "RX"}:
            reset.update(qs)
            key = "reset"
        elif name in {"CX", "CZ"}:
            key = "2q"
        else:
            one.update(qs)
            key = "1q"
        duration = max(duration, calibration.durations_ns[key] if calibration else 1)
    finish_moment()
    candidates, seen = [], set()
    for orientation in range(8):
        trans = {}
        for q, (u, v) in positions.items():
            if orientation & 4:
                u, v = v, u
            trans[q] = (u * (-1 if orientation & 1 else 1), v * (-1 if orientation & 2 else 1))
        minr, minc = min(r for r, c in trans.values()), min(c for r, c in trans.values())
        trans = {q: (r-minr, c-minc) for q, (r, c) in trans.items()}
        height, width = max(r for r, c in trans.values())+1, max(c for r, c in trans.values())+1
        for row in range(rows-height+1):
            for col in range(cols-width+1):
                mapping = {q: grid[row+r][col+c] for q, (r, c) in trans.items()}
                if None in mapping.values():
                    continue
                signature = tuple(mapping[q] for q in sorted(mapping))
                if signature in seen:
                    continue
                seen.add(signature)
                try:
                    validate_layout(circuit, mapping, couplings)
                    score = 0.0
                    if calibration:
                        for q, physical in mapping.items():
                            cal = calibration.qubits[physical]
                            score += one[q] * -math.log1p(-cal.single_qubit_infidelity)
                            score += readout_weight * measure[q] * -math.log1p(-cal.readout_error)
                            score += reset[q] * -math.log1p(-cal.reset_error)
                            score += coherence_weight * idle_ns[q] * (1/cal.t1_ns + 1/cal.t2_ns)
                            touches = sum(n for pair, n in pair_counts.items() if q in pair)
                            score += leakage_weight * touches * cal.leakage_per_gate
                        score += sum(n * -math.log1p(-calibration.edge_error(mapping[a], mapping[b]))
                                     for (a, b), n in pair_counts.items())
                except (ValueError, KeyError):
                    continue
                candidates.append(Placement(mapping, score, orientation, (row, col)))
    if not candidates:
        raise ValueError("No native placement in supplied grid/symmetry family. Check topology/size; qubit count alone is insufficient.")
    return sorted(candidates, key=lambda p: (p.score, p.orientation, p.origin))


def build_emerald_qubit_map(stim_circuit, grid_top_left=41, grid_n_rows=5,
                           grid_n_cols=5, row_stride=-8, col_stride=1, qiskit_offset=-1):
    """Compatibility wrapper: dense Qiskit index -> physical Qiskit index."""
    grid = [[f"QB{grid_top_left+r*row_stride+c*col_stride}" for c in range(grid_n_cols)]
            for r in range(grid_n_rows)]
    placement = rank_placements(stim_circuit, grid=grid)[0]
    return {i: int(placement.stim_to_physical[q][2:])+qiskit_offset
            for i, q in enumerate(sorted(placement.stim_to_physical))}


def print_qubit_map(stim_circuit, **kwargs):
    mapping = build_emerald_qubit_map(stim_circuit, **kwargs)
    print("Dense index -> physical Qiskit index:", mapping)
    print("Geometric placement: all circuit interactions native; verify live backend couplings before submission.")
    return mapping
