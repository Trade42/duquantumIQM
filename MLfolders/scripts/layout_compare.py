"""Zero-SWAP placement of the d=3 rotated surface code on IQMFakeAphrodite, proven against Qiskit's automatic layout.

Prints, and writes to data/sim/layout/compare.json and compare.md:

1. the device grid, inferred from its couplers;
2. every valid d=3 placement, cross-checked against networkx's subgraph monomorphisms, and
   whether a d=5 placement exists;
3. the organisers' 5x5 example (challenge/build_emerald_qubit_rotated.py) checked against the couplers;
4. the comparison table: ours (our placement, our native gates; with a barrier at every TICK,
   the default, and without) against Qiskit's automatic layout at optimization levels 1 and 3,
   d=3, rounds 1, 3, 5, bases Z and X;
5. the equivalence checks: exact (Stim) for every circuit, every single Pauli X or Z after
   every TICK of the source (rounds 3), and the noiseless Aer measurement-order run.

The source circuit is the contract stand-in (layout.standin_circuit): Stim's generated with-reset
circuit, X basis rewritten to R+H / H+M. Gate counts use its MR -> M form, the hardware target.

Run: uv run python scripts/layout_compare.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import stim
from iqm.qiskit_iqm import IQMFakeAphrodite
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from dqiqm import hw
from dqiqm import layout as L

SEED = L.SEED
ROUNDS = (1, 3, 5)
BASES = ("Z", "X")
AER_SHOTS = 200
OUT = Path(__file__).resolve().parents[1] / "data" / "sim" / "layout"

# The organisers' docstring example: Stim qubit -> (row, col) of their 5x5 block,
# QB = 41 - 8*row + col (challenge/build_emerald_qubit_rotated.py, build_emerald_qubit_map).
ORGANISERS_5X5 = {
    5: (0, 2), 13: (0, 3),
    2: (1, 0), 3: (1, 1), 11: (1, 2), 12: (1, 3),
    1: (2, 0), 9: (2, 1), 10: (2, 2), 18: (2, 3), 19: (2, 4),
    8: (3, 1), 16: (3, 2), 17: (3, 3), 25: (3, 4),
    14: (4, 1), 15: (4, 2),
}  # fmt: skip


def plain_circuit(source: stim.Circuit) -> QuantumCircuit:
    """The logical circuit Qiskit gets: the organisers' stim_to_qiskit (challenge/stim_to_qiskit.py) as h, cx, measure.

    Dense qubits in ascending Stim order, measurement j in clbit j. TICKs are dropped (no
    barriers), so Qiskit may merge single-qubit gates across layers, its best case. The initial
    R is dropped (qubits start in |0>; the IQM target has no reset). Hardware form: no MR.
    """
    flat = source.flattened()
    qubits = L._gate_qubits(flat)
    dense = {q: i for i, q in enumerate(qubits)}
    qc = QuantumCircuit(len(qubits), flat.num_measurements)
    j = 0
    for inst in flat:
        ts = [dense[t.value] for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name == "H":
            for i in ts:
                qc.h(i)
        elif inst.name == "CX":
            for c, t in zip(ts[::2], ts[1::2]):
                qc.cx(c, t)
        elif inst.name == "M":
            for i in ts:
                qc.measure(i, j)
                j += 1
        elif inst.name not in ("R", "TICK", *L._ANNOTATIONS):
            raise ValueError(inst.name)
    return qc


def readout_layers(qc: QuantumCircuit) -> int:
    """How many distinct ASAP layers hold a measurement (barriers synchronise). Fewer means more parallel readout."""
    level: dict[int, int] = {}
    layers = set()
    for ci in qc.data:
        qs = [qc.find_bit(q).index for q in ci.qubits]
        top = max((level.get(q, 0) for q in qs), default=0)
        if ci.operation.name != "barrier":
            top += 1
            if ci.operation.name == "measure":
                layers.add(top)
        for q in qs:
            level[q] = top
    return len(layers)


def stats(qc: QuantumCircuit, rounds: int, backend) -> dict:
    ops = qc.count_ops()
    try:
        duration = qc.estimate_duration(backend.target, unit="s")
    except Exception:  # the fake backends carry no gate durations
        duration = None
    return {
        "cz": ops.get("cz", 0),
        "extra_cz": ops.get("cz", 0) - 24 * rounds,
        "cz_per_round": ops.get("cz", 0) / rounds,
        "r": ops.get("r", 0),
        "r_per_round": ops.get("r", 0) / rounds,
        "measure": ops.get("measure", 0),
        "depth": qc.depth(lambda ci: ci.operation.name != "barrier"),
        "cz_depth": qc.depth(lambda ci: ci.operation.name == "cz"),
        "readout_layers": readout_layers(qc),
        "duration_s": duration,
    }


def qiskit_auto(source: stim.Circuit, backend, level: int) -> tuple[QuantumCircuit, int, list[int]]:
    """Qiskit's own layout and routing: (transpiled circuit, SWAPs its router inserted, its initial layout)."""
    swaps = [0]

    def watch(**kw):
        swaps[0] = max(swaps[0], kw["dag"].count_ops().get("swap", 0))

    t = transpile(plain_circuit(source), backend=backend, optimization_level=level, seed_transpiler=SEED, callback=watch)
    return t, swaps[0], list(t.layout.initial_index_layout(filter_ancillas=True))


def on_couplers(qc: QuantumCircuit, backend, arch: L.Architecture) -> bool:
    names = {i: backend.index_to_qubit_name(i) for i in range(backend.num_qubits)}
    return all(arch.coupled(*i.locus) for i in L.sent_instructions(qc, names) if i.name == "cz")


def organisers_placement(arch: L.Architecture, placements: list[L.Placement]) -> dict:
    mapping = {q: f"QB{41 - 8 * r + c}" for q, (r, c) in ORGANISERS_5X5.items()}
    src = L.standin_circuit(3, 1, "Z")
    bad = [(a, b) for a, b in L.cx_pairs(src) if not arch.coupled(mapping[a], mapping[b])]
    match = [p for p in placements if p.physical == mapping]
    return {
        "mapping": {str(k): v for k, v in sorted(mapping.items())},
        "cx_pairs_off_coupler": len(bad),
        "is_one_of_ours": bool(match),
        "origin": list(match[0].origin) if match else None,
        "orientation": match[0].orientation if match else None,
    }


def monomorphisms(source: stim.Circuit, arch: L.Architecture, limit_s: float = 120) -> dict:
    """Every injective map of the code's CX graph into the coupler graph, by networkx (independent of our grid)."""
    hwg = nx.Graph([tuple(c) for c in arch.couplers])
    code = nx.Graph(set(L.cx_pairs(source)))
    t0 = time.time()
    count, sets, complete = 0, set(), True
    for m in nx.algorithms.isomorphism.GraphMatcher(hwg, code).subgraph_monomorphisms_iter():
        count += 1
        sets.add(frozenset(m))
        if time.time() - t0 > limit_s:
            complete = False
            break
    auts = sum(1 for _ in nx.algorithms.isomorphism.GraphMatcher(code, code).isomorphisms_iter())
    return {"monomorphisms": count, "qubit_sets": len(sets), "code_automorphisms": auts, "complete": complete}


def fault_sweep(source: stim.Circuit) -> dict:
    """Every single X or Z after every TICK of the source, on every qubit: converted, read back into Stim, compared.

    The converted circuit's detection events (through the source's m2d converter) must equal the
    faulty source's own, on a noiseless shot. Faults are deterministic, so one shot decides.
    """
    m2d = source.compile_m2d_converter()
    qubits = L._gate_qubits(source)
    ticks = sum(1 for i in source.flattened() if i.name == "TICK")
    cases = mismatches = fired = 0
    for tick in range(ticks):
        for q in qubits:
            for pauli in ("X", "Z"):
                faulty = L.insert_pauli(source, tick, q, pauli)
                want = m2d.convert(measurements=faulty.compile_sampler(seed=SEED).sample(1), separate_observables=True)
                native, _ = L.native_circuit(faulty, target="simulation")
                rt, clbits = L.native_to_stim(native, source)
                sample = rt.compile_sampler(seed=SEED).sample(1)
                record = np.zeros_like(sample)
                record[:, clbits] = sample
                got = m2d.convert(measurements=record, separate_observables=True)
                cases += 1
                fired += bool(want[0].any() or want[1].any())
                mismatches += not (np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1]))
    return {"cases": cases, "cases_that_fire": fired, "mismatches": mismatches}


def aer_check(source: stim.Circuit, backend, placement: L.Placement) -> dict:
    hc = L.to_hardware(source, backend, placement, target="simulation")
    res = AerSimulator(seed_simulator=SEED).run(hc.circuit, shots=AER_SHOTS, memory=True).result()
    bits = hw._memory_bits(res.get_memory())  # the same column order hw.save_raw stores
    det, obs = source.compile_m2d_converter().convert(measurements=bits.astype(bool), separate_observables=True)
    return {
        "shots": AER_SHOTS,
        "measurements": int(bits.shape[1]),
        "detection_events": int(det.sum()),
        "observable_flips": int(obs.sum()),
        "random_columns": int((bits.min(axis=0) != bits.max(axis=0)).sum()),
    }


def same_gates(hardware: L.HardwareCircuit, simulation: L.HardwareCircuit) -> bool:
    """The hardware circuit is the simulation circuit without its resets, qubit for qubit."""
    a = L._per_qubit(L.sent_instructions(hardware.circuit, hardware.qubit_index_to_name))
    b = L._per_qubit(
        i for i in L.sent_instructions(simulation.circuit, simulation.qubit_index_to_name) if i.name != "reset"
    )
    return a == b


def fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def main() -> None:
    t_start = time.time()
    backend = IQMFakeAphrodite()
    arch = L.Architecture.from_backend(backend)
    print("IQMFakeAphrodite: 54 qubits, 90 couplers; grid inferred from the couplers:\n")
    print(arch.grid(), "\n")
    out: dict = {"device": "IQMFakeAphrodite", "seed": SEED, "qubits": len(arch.qubits), "couplers": len(arch.couplers)}

    # --- placements -------------------------------------------------------------------------
    d3 = L.standin_circuit(3, 1, "Z")
    d5 = L.standin_circuit(5, 1, "Z")
    p3, p5 = L.placements(d3, arch), L.placements(d5, arch)
    nx3, nx5 = monomorphisms(d3, arch), monomorphisms(d5, arch)
    out["placements"] = {
        "d3": {"placements": len(p3), "qubit_sets": len({p.qubit_set for p in p3}), "networkx": nx3},
        "d5": {
            "placements": len(p5),
            "qubit_sets": len({p.qubit_set for p in p5}),
            "networkx": nx5,
            "list": [{"origin": list(p.origin), "orientation": p.orientation} for p in p5],
        },
    }
    print(f"d=3: {len(p3)} placements on {len({p.qubit_set for p in p3})} distinct qubit sets "
          f"(networkx: {nx3['monomorphisms']} monomorphisms, {nx3['qubit_sets']} sets, "
          f"{nx3['code_automorphisms']} automorphisms of the code's CX graph)")
    print(f"d=5: {len(p5)} placements on {len({p.qubit_set for p in p5})} distinct qubit sets "
          f"(networkx: {nx5['monomorphisms']} monomorphisms{'' if nx5['complete'] else ', search cut short'})")
    if p5:
        print("\nd=5, first placement (origin, orientation) =", p5[0].origin, p5[0].orientation)
        print(L.placement_grid(p5[0], arch, d5))
    assert nx3["complete"] and nx3["monomorphisms"] == len(p3), "the grid enumeration missed a placement"
    assert nx5["complete"] and nx5["monomorphisms"] == len(p5), "the grid enumeration missed a d=5 placement"

    org = organisers_placement(arch, p3)
    out["organisers_5x5"] = org
    print(f"\nThe organisers' 5x5 example: {org['cx_pairs_off_coupler']} of 24 CX pairs off a coupler; "
          f"one of ours: {org['is_one_of_ours']} (origin {org['origin']}, orientation {org['orientation']})")
    placement = L.place(d3, arch, tuple(org["origin"]), org["orientation"]) if org["is_one_of_ours"] else p3[0]
    out["placement"] = {
        "origin": list(placement.origin),
        "orientation": placement.orientation,
        "mapping": {str(q): n for q, n in placement.mapping},
    }
    print("\nThe placement used below:")
    print(L.placement_grid(placement, arch, d3), "\n")

    # --- the comparison --------------------------------------------------------------------
    rows = []
    for basis in BASES:
        for rounds in ROUNDS:
            hw_src = L.standin_circuit(3, rounds, basis, reset=False)
            for method, barriers in (("ours", True), ("ours, no barriers", False)):
                ours = L.to_hardware(hw_src, backend, placement, barriers=barriers)  # check_hardware: 0 SWAPs
                row = {"basis": basis, "rounds": rounds, "method": method, "swaps": 0, "on_couplers": True}
                row |= stats(ours.circuit, rounds, backend)
                row["layout"] = list(ours.initial_layout)
                rows.append(row)
            for level in (1, 3):
                t, swaps, lay = qiskit_auto(hw_src, backend, level)
                row = {"basis": basis, "rounds": rounds, "method": f"qiskit L{level}", "swaps": swaps}
                row["on_couplers"] = on_couplers(t, backend, arch)
                row |= stats(t, rounds, backend)
                row["layout"] = lay
                rows.append(row)
    out["comparison"] = rows

    cols = ["basis", "rounds", "method", "swaps", "extra_cz", "cz", "cz_per_round", "r", "r_per_round",
            "measure", "depth", "cz_depth", "readout_layers", "duration_s"]
    head = "| " + " | ".join(cols) + " |"
    table = [head, "|" + "---|" * len(cols)]
    table += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    print("\n".join(table))
    steady = {}
    for method in ("ours", "ours, no barriers", "qiskit L1", "qiskit L3"):
        for basis in BASES:
            r3 = next(r for r in rows if (r["method"], r["basis"], r["rounds"]) == (method, basis, 3))
            r5 = next(r for r in rows if (r["method"], r["basis"], r["rounds"]) == (method, basis, 5))
            keys = ("cz", "r", "measure", "depth", "cz_depth", "readout_layers")
            steady[f"{method} {basis}"] = {k: (r5[k] - r3[k]) / 2 for k in keys}
    out["per_round_steady_state"] = steady
    print("\nPer extra round (rounds 3 -> 5, halved):")
    for k, v in steady.items():
        print(f"  {k:20s} " + ", ".join(f"{n} {fmt(x)}" for n, x in v.items()))
    out["qiskit_layouts"] = {
        f"{r['method']} {r['basis']} r{r['rounds']}": sorted(backend.index_to_qubit_name(i) for i in r["layout"])
        for r in rows
        if r["method"].startswith("qiskit")
    }

    # --- equivalence ------------------------------------------------------------------------
    checks = {}
    for basis in BASES:
        for rounds in ROUNDS:
            src = L.standin_circuit(3, rounds, basis)
            sim = L.to_hardware(src, backend, placement, target="simulation")
            hard = L.to_hardware(L.standin_circuit(3, rounds, basis, reset=False), backend, placement)
            L.check_equivalent(sim.circuit, src)  # raises on failure
            nobar = L.to_hardware(src, backend, placement, target="simulation", barriers=False)
            L.check_equivalent(nobar.circuit, src)
            checks[f"{basis} r{rounds}"] = {
                "exact_stim": "pass",
                "exact_stim_no_barriers": "pass",
                "hardware_equals_simulation_without_resets": same_gates(hard, sim),
                "aer": aer_check(src, backend, placement),
            }
    for basis in BASES:
        checks[f"{basis} r3 faults"] = fault_sweep(L.standin_circuit(3, 3, basis))
    out["equivalence"] = checks
    print("\nEquivalence (source: with-reset stand-in, simulation target):")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    # d=5 on the one region it fits
    if p5:
        d5_rows = {}
        for rounds in (1, 3):
            src5 = L.standin_circuit(5, rounds, "Z")
            hc5 = L.to_hardware(L.standin_circuit(5, rounds, "Z", reset=False), backend, p5[0])
            L.check_equivalent(L.to_hardware(src5, backend, p5[0], target="simulation").circuit, src5)
            s = stats(hc5.circuit, rounds, backend)
            d5_rows[f"Z r{rounds}"] = {k: s[k] for k in ("cz", "r", "measure", "depth", "cz_depth")} | {"exact_stim": "pass"}
        out["d5"] = d5_rows
        print("\nd=5 (80 CZ per round), first placement:", d5_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(out, indent=2) + "\n")
    (OUT / "compare.md").write_text(
        "# d=3 on IQMFakeAphrodite: ours against Qiskit's automatic layout\n\n"
        "Written by `scripts/layout_compare.py`; the full record is `compare.json`.\n\n" + "\n".join(table) + "\n"
    )
    print(f"\nwrote {OUT / 'compare.json'} and compare.md in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
