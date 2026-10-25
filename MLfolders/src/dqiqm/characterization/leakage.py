"""C3, CZ leakage: a design in IQM native instructions, and its readout algebra. Not runnable today.

Why it cannot run
    It needs ``prx_12`` (the |1>-|2> drive), which Garnet calibrates on all 20 qubits
    (``modulated_drag_crf``) and iqm-client 35.0.3 accepts as a native operation, but our circuit
    path cannot send it: qiskit has no gate that the IQM serializer turns into ``prx_12`` (its
    ``allowed_nonnative_gates`` would write the arguments as ``p0``/``p1``, not ``angle``/``phase``),
    and ``dqiqm.hw.submit`` takes qiskit circuits only. ``measure_012`` (calibrated on QB1, QB3,
    QB9 and QB14) is not an operation iqm-client 35.0.3 knows at all. ``docs/garnet-calib.md`` lists
    what ``hw.py`` would need.

The experiment, per coupler (f, o)
    f is the qubit whose flux pulse makes the CZ: the first qubit of IQM's calibration locus (for
    example QB4 in ``QB4__QB1``), on this patch always the data qubit. A flux CZ leaks through
    the |11> <-> |20> crossing of the fluxed qubit, so the leaked population is f's |2>. That is
    the physics of the gate family, inferred, not measured. Both qubits are prepared in |1>
    (two ``prx(pi/2, 0)`` each), n CZs run (n in :data:`C3_N`, a barrier after each), and f is read
    out three ways: ``plain`` (just ``measure``), ``swap12`` (``prx_12(pi)`` first, |1> <-> |2>) and
    ``flip01`` (an X first, |0> <-> |1>). Three references fix how the two-state discriminator reads
    each level: ``ref0`` (|0>), ``ref1`` (X) and ``ref2`` (X then ``prx_12(pi)``, |2>).

The algebra (:func:`populations`)
    With r_k = P(read 1 | level k) from the references and populations (p0, p1, p2),

        plain  = r0 p0 + r1 p1 + r2 p2
        swap12 = r0 p0 + r1 p2 + r2 p1
        flip01 = r0 p1 + r1 p0 + r2 p2,      p0 + p1 + p2 = 1,

    solved by least squares. It needs r2 away from (r0 + r1)/2 (the |2> blob not halfway between
    the |0> and |1> ones) or r2 away from r1; ``ref2`` measures that before anything is fitted.
    The leakage per CZ is the slope of p2 against n. Where f has ``measure_012`` (QB14 on this
    patch: QB14-QB9, QB14-QB13, QB14-QB15), a direct three-level readout is the cross-check.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dqiqm.experiments import device as dev

C3_N = (0, 1, 3, 9)
VARIANTS = ("plain", "swap12", "flip01")
REFERENCES = ("ref0", "ref1", "ref2")
MEASURE_012 = ("QB1", "QB3", "QB9", "QB14")  # the snapshot's measure_012 loci


def fluxed(cz_loci: Sequence[str], a: str, b: str) -> str:
    """The first qubit of the coupler's calibration locus (``QBx__QBy`` keys of the pulse file)."""
    for key in cz_loci:
        first, second = key.split("__")
        if {first, second} == {a, b}:
            return first
    raise KeyError(f"no CZ calibration locus for {a}-{b}")


def _op(name: str, locus: Sequence[str], **args: Any) -> dict[str, Any]:
    return {"name": name, "locus": list(locus), "args": args}


def _x(q: str) -> list[dict[str, Any]]:
    return [_op("prx", [q], angle=math.pi / 2, phase=0.0), _op("barrier", [q]),
            _op("prx", [q], angle=math.pi / 2, phase=0.0), _op("barrier", [q])]


def circuit(f: str, o: str, kind: str, n: int = 0) -> dict[str, Any]:
    """One C3 circuit as IQM ``CircuitOperation`` dicts (name, locus, args): ``kind`` is a variant
    (with n CZs) or a reference (no CZ). Measurement keys are ``m_<qubit>``."""
    pair = [f, o]
    ops: list[dict[str, Any]] = []
    if kind in VARIANTS:
        ops += [_op("prx", [q], angle=math.pi / 2, phase=0.0) for q in pair] + [_op("barrier", pair)]
        ops += [_op("prx", [q], angle=math.pi / 2, phase=0.0) for q in pair] + [_op("barrier", pair)]
        for _ in range(n):
            ops += [_op("cz", pair), _op("barrier", pair)]
        if kind == "swap12":
            ops += [_op("prx_12", [f], angle=math.pi, phase=0.0), _op("barrier", pair)]
        elif kind == "flip01":
            ops += _x(f)
        name = f"C3_{f}-{o}_{kind}_n{n:02d}"
    elif kind in REFERENCES:
        if kind in ("ref1", "ref2"):
            ops += _x(f)
        if kind == "ref2":
            ops += [_op("prx_12", [f], angle=math.pi, phase=0.0), _op("barrier", pair)]
        name = f"C3_{f}-{o}_{kind}"
    else:
        raise ValueError(kind)
    ops += [_op("measure", [q], key=f"m_{q}") for q in pair]
    return {"name": name, "fluxed": f, "other": o, "kind": kind, "n": n, "instructions": ops}


def design(couplers: Sequence[tuple[str, str]], cz_loci: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for a, b in couplers:
        f = fluxed(cz_loci, a, b)
        o = b if f == a else a
        out += [circuit(f, o, k) for k in REFERENCES]
        out += [circuit(f, o, k, n) for n in C3_N for k in VARIANTS]
    return out


def populations(m: Mapping[str, float], r: Sequence[float]) -> np.ndarray:
    """(p0, p1, p2) from the three readouts ``m`` (plain, swap12, flip01: P(read 1)) and the
    references r = (r0, r1, r2), by least squares with p0 + p1 + p2 = 1 as a fourth equation."""
    r0, r1, r2 = r
    a = np.array([[r0, r1, r2], [r0, r2, r1], [r1, r0, r2], [1.0, 1.0, 1.0]])
    y = np.array([m["plain"], m["swap12"], m["flip01"], 1.0])
    p, *_ = np.linalg.lstsq(a, y, rcond=None)
    return p


def readouts(p: Sequence[float], r: Sequence[float]) -> dict[str, float]:
    """The forward map of :func:`populations` (for its test)."""
    p0, p1, p2 = p
    r0, r1, r2 = r
    return {"plain": r0 * p0 + r1 * p1 + r2 * p2, "swap12": r0 * p0 + r1 * p2 + r2 * p1,
            "flip01": r0 * p1 + r1 * p0 + r2 * p2}


def leakage_per_cz(n: Sequence[int], p2: Sequence[float], se: Sequence[float]) -> tuple[float, float]:
    """The slope of p2 against n (weighted least squares) and its standard error."""
    x = np.column_stack([np.ones(len(n)), np.asarray(n, float)])
    w = 1 / np.asarray(se, float) ** 2
    cov = np.linalg.inv((x.T * w) @ x)
    beta = cov @ ((x.T * w) @ np.asarray(p2, float))
    return float(beta[1]), float(math.sqrt(cov[1, 1]))


def summary(couplers: Sequence[tuple[str, str]], cz_loci: Sequence[str]) -> dict[str, Any]:
    rows = {}
    for a, b in couplers:
        f = fluxed(cz_loci, a, b)
        rows[f"{a}-{b}"] = {"fluxed": f, "measure_012_on_fluxed": f in MEASURE_012}
    return {"couplers": rows, "n": list(C3_N), "variants": list(VARIANTS), "references": list(REFERENCES),
            "per_coupler_circuits": len(REFERENCES) + len(VARIANTS) * len(C3_N),
            "status": "blocked: prx_12 cannot go through dqiqm.hw.submit; measure_012 is not in iqm-client 35.0.3",
            "device": "garnet", "sorted_pairs": [list(dev.pair_key(a, b)) for a, b in couplers]}
