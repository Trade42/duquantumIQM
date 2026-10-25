"""E1 and E4: the d=3 no-reset surface-code memory on Garnet, and the same with idles between rounds.

The circuit is ``dqiqm.surface.memory_circuit(3, R, basis, no_reset=True)``, unchanged: its
detectors are, in order and coordinates, those of the team's ``make_stim_circuit`` (checked by the
orchestrator, 2026-10-24 19:07 UTC), so the ML pipeline that uses the team's generator reads these
shots as they are.

Placement
    On Garnet the code fits with zero SWAPs on exactly one set of 17 qubits (all but QB3, QB17 and
    QB18), in 4 placements: the code graph's 4 automorphisms. Two of them make {QB1, QB9, QB11,
    QB20} the X-type ancillas, two make {QB5, QB7, QB13, QB15} X-type; within a pair, the two differ
    by a half turn, which moves the logical operator and the CX order but keeps every qubit's role.
    So there is no second patch. The record's "patches" are:

    * ``pA``: the placement with the lowest P_L under the calibrated Stim noise (:func:`rank`);
    * ``pB``: the best placement whose X-type ancillas are the other set, so every ancilla changes
      its check type. It separates qubit effects from check-type effects, which is the only spatial
      variation Garnet allows.

E4
    The same circuit with ``delay(t)`` on every qubit after each round's readout but the last
    (R − 1 idles), t in {1, 2, 5} µs; t = 0 is E1. The idles let T1 (Z basis, through the ancillas
    and the data's X errors) and T2 (X basis) act. IQMFakeGarnet puts no noise on ``delay``, so
    offline E4 is checked for structure only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any

import numpy as np
import stim

from dqiqm import decode as D
from dqiqm import layout as L
from dqiqm import surface as S
from dqiqm.experiments import device as dev
from dqiqm.experiments.circuits import CircuitSpec, Stabilizer, insert_round_delays, place

E1_ROUNDS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
E4_ROUNDS = (2, 3, 5, 8)
E4_DELAYS_NS = (1000.0, 2000.0, 5000.0)
BASES = ("Z", "X")


def stabilizers(g: S.SurfaceGeometry) -> dict[int, Stabilizer]:
    out = {}
    for a in g.ancillas:
        pauli = "X" if a in g.x_ancillas else "Z"
        x, y = g.coords[a]
        out[a] = Stabilizer(f"{pauli}({x},{y})", pauli, tuple(g.support[a]))
    return out


def roles(g: S.SurfaceGeometry) -> dict[int, str]:
    return {q: ("data" if q in g.data else "ancilla") for q in g.coords}


def meas_rounds(g: S.SurfaceGeometry, rounds: int) -> tuple[int, ...]:
    return tuple([k for k in range(1, rounds + 1) for _ in g.ancillas] + [rounds + 1] * len(g.data))


def cx_pairs(g: S.SurfaceGeometry) -> tuple[tuple[int, int], ...]:
    return tuple(pair for layer in g.cx_layers for pair in layer)


def placements(arch: L.Architecture) -> list[L.Placement]:
    return L.placements(S.memory_circuit(3, 2, "Z"), arch)


def rank(arch: L.Architecture, cal: dev.Calibration, *, shots: int = 20_000, rounds: int = 3, seed: int = 2026) -> list[dict]:
    """Every placement, scored by P_L under the calibrated Stim noise (Z and X, ``rounds`` rounds), best first."""
    g = S.geometry(3)
    out = []
    for i, p in enumerate(placements(arch)):
        nm = dev.calibrated_noise(cal, p.physical, roles(g), pairs=cx_pairs(g), x_ancillas=g.x_ancillas)
        pl = {}
        for j, basis in enumerate(BASES):
            c = S.memory_circuit(3, rounds, basis, no_reset=True, noise=nm)
            dets, obs = c.compile_detector_sampler(seed=seed + 10 * i + j).sample(shots, separate_observables=True)
            pred = D.decode(c, dets)
            pl[basis] = D.logical_error_probability(pred, obs).p
        x_set = tuple(sorted((p.physical[a] for a in g.x_ancillas), key=dev.qubit_sort_key))
        out.append({"placement": p, "p_l": pl, "mean_p_l": float(np.mean(list(pl.values()))), "x_ancillas": x_set})
    out.sort(key=lambda r: (r["mean_p_l"], r["placement"].orientation))
    return out


def choose_patches(ranked: Sequence[dict]) -> dict[str, dict]:
    """``pA``: the best placement; ``pB``: the best one with the other X-type ancilla set."""
    best = ranked[0]
    other = next((r for r in ranked if r["x_ancillas"] != best["x_ancillas"]), None)
    return {"pA": best} | ({"pB": other} if other is not None else {})


def memory_noisy(rounds: int, basis: str, noise: Any) -> stim.Circuit:
    return S.memory_circuit(3, rounds, basis, no_reset=True, noise=noise)


def spec(
    rounds: int,
    basis: str,
    placement: L.Placement,
    patch: str,
    backend: Any,
    *,
    delay_ns: float = 0.0,
    barriers: bool = True,
    plain: L.HardwareCircuit | None = None,
) -> CircuitSpec:
    """One E1 (``delay_ns=0``) or E4 circuit on ``placement``."""
    g = S.geometry(3)
    source = S.memory_circuit(3, rounds, basis, no_reset=True)
    experiment = "E1" if delay_ns == 0 else "E4"
    name = f"{experiment}_d3_{basis}_r{rounds:02d}_{patch}"
    if delay_ns:
        name += f"_t{int(round(delay_ns)):05d}ns"
    if not barriers:
        name += "_nobar"
    native, stim_qubits = L.native_circuit(source, target="hardware", barriers=barriers, name=name)
    checks: dict[str, Any] = {}
    if delay_ns and rounds > 1:
        if plain is None:
            plain = place(source, native, stim_qubits, backend, placement)
        delayed = insert_round_delays(native, delay_ns, range(1, rounds))
        hc = place(source, delayed, stim_qubits, backend, placement, plain=plain, delays=(delay_ns, rounds - 1))
        checks["delays"] = {"per_qubit": rounds - 1, "delay_ns": delay_ns, "plain_checked": True}
    else:
        hc = place(source, native, stim_qubits, backend, placement)
    checks["swaps"] = int(hc.circuit.count_ops().get("swap", 0))
    return CircuitSpec(
        experiment=experiment,
        name=name,
        stim=source,
        hardware=hc,
        params={
            "family": "surface_memory",
            "distance": 3,
            "rounds": rounds,
            "basis": basis,
            "patch": patch,
            "orientation": placement.orientation,
            "delay_ns": float(delay_ns),
            "barriers": barriers,
            "prep": "0" if basis == "Z" else "+",
        },
        roles=roles(g),
        stabilizers=stabilizers(g),
        meas_round=meas_rounds(g, rounds),
        noisy=partial(memory_noisy, rounds, basis),
        x_ancillas=tuple(g.x_ancillas),
        cx_pairs=cx_pairs(g),
        checks=checks,
    )


def e1_specs(patches: dict[str, L.Placement], backend: Any, rounds: Iterable[int] = E1_ROUNDS) -> list[CircuitSpec]:
    return [spec(r, b, p, name, backend) for name, p in patches.items() for b in BASES for r in rounds]


def e4_specs(
    placement: L.Placement,
    backend: Any,
    patch: str = "pA",
    rounds: Iterable[int] = E4_ROUNDS,
    delays_ns: Iterable[float] = E4_DELAYS_NS,
) -> list[CircuitSpec]:
    out = []
    for b in BASES:
        for r in rounds:
            plain = spec(r, b, placement, patch, backend).hardware  # the E1 circuit, checked by check_hardware
            out.extend(spec(r, b, placement, patch, backend, delay_ns=t, plain=plain) for t in delays_ns)
    return out
