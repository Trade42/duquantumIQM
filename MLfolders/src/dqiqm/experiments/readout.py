"""E3: readout characterisation on every Garnet qubit at once.

Two kinds of circuit, all 20 qubits read in parallel in each readout layer (as the codes read
their ancillas), so crosstalk and correlated misassignment show up as they do in the codes:

* **assignment** (one readout): all |0>, all |1>, and 16 seeded random computational states.
  Per qubit they give P(1|0) and P(0|1); across qubits, the correlation of misassignments
  (correlated readout errors), and the dependence of one qubit's error on its neighbours' states;
* **repeated readout without reset** (15 readouts in a row): all |0>, all |1>, and the two
  checkerboards. Detectors D_1 = m_1 and D_k = m_k + m_(k−1), against the prepared bit:

  - a record flip (the bit is misread, the qubit is not flipped) at readout k fires D_k and
    D_(k+1): a pair, adjacent in time;
  - a state flip between readouts k−1 and k (T1 decay during the readout window, or a
    measurement-induced flip) fires D_k alone, and every later bit stays flipped.

  This separates the two, which the surface-gen finding needs: without reset, a record flip
  fires D_k and D_(k+2) in the code, a state flip D_(k+1) and D_(k+2).

The X of a |1> is ``r(pi, 0)``, prepended to the converted X-free source (the converter has no
X). Stim qubit i is the backend's physical qubit i; QUBIT_COORDS are its grid (col, row).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import stim

from dqiqm import layout as L
from dqiqm.experiments.circuits import CircuitSpec, Stabilizer, place, prepend_x

N_RANDOM = 16
REPEATS = 15
SEED = 2026


def circuit(coords: Sequence[tuple[int, int]], prep: Sequence[int], repeats: int, *,
            record_flip: float = 0.0, state_flip: float = 0.0, with_x: bool = True) -> stim.Circuit:
    """``repeats`` readouts of every qubit after preparing the bits ``prep``.

    ``record_flip`` writes ``M(p)``; ``state_flip`` an ``X_ERROR(p)`` between readouts. Both are for
    Stim checks of the analysis only: the recorded circuit has neither.
    """
    n = len(prep)
    c = stim.Circuit()
    for q, (row, col) in enumerate(coords):
        c.append("QUBIT_COORDS", [q], [col, row])
    c.append("R", range(n))
    ones = [q for q in range(n) if prep[q]]
    if ones and with_x:
        c.append("X", ones)
    for k in range(1, repeats + 1):
        c.append("TICK")
        if k > 1 and state_flip > 0:
            c.append("X_ERROR", range(n), state_flip)
        c.append("M", range(n), record_flip) if record_flip > 0 else c.append("M", range(n))
        for q in range(n):
            row, col = coords[q]
            refs = [stim.target_rec(q - n)] + ([stim.target_rec(q - 2 * n)] if k > 1 else [])
            c.append("DETECTOR", refs, [col, row, k - 1])
    return c


def spec(name: str, prep: Sequence[int], repeats: int, backend: Any, arch: L.Architecture, kind: str) -> CircuitSpec:
    qubits = [backend.index_to_qubit_name(i) for i in range(backend.num_qubits)]
    coords = [arch.coords[q] for q in qubits]
    source = circuit(coords, prep, repeats)
    native, stim_qubits = L.native_circuit(circuit(coords, prep, repeats, with_x=False), target="hardware", name=name)
    native = prepend_x(native, [stim_qubits.index(q) for q in range(len(prep)) if prep[q]])
    placement = L.Placement(origin=(0, 0), orientation=0, mapping=tuple(enumerate(qubits)))
    hc = place(source, native, stim_qubits, backend, placement)
    n = len(prep)
    return CircuitSpec(
        experiment="E3",
        name=name,
        stim=source,
        hardware=hc,
        params={
            "family": "readout",
            "kind": kind,
            "rounds": repeats,
            "basis": "Z",
            "prep": "".join(str(int(b)) for b in prep),
            "patch": "all20",
            "delay_ns": 0.0,
            "barriers": True,
        },
        roles={q: "readout" for q in range(n)},
        stabilizers={q: Stabilizer(f"M({qubits[q]})", "Z", (q,)) for q in range(n)},
        meas_round=tuple(k for k in range(1, repeats + 1) for _ in range(n)),
        noisy=None,
        checks={"swaps": int(hc.circuit.count_ops().get("swap", 0))},
    )


def preps(arch: L.Architecture, qubits: Sequence[str], seed: int = SEED, n_random: int = N_RANDOM) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    n = len(qubits)
    out = {"all0": [0] * n, "all1": [1] * n}
    for i in range(n_random):
        out[f"rand{i:02d}"] = [int(b) for b in rng.integers(0, 2, n)]
    parity = [sum(arch.coords[q]) % 2 for q in qubits]
    out["chkA"] = parity
    out["chkB"] = [1 - p for p in parity]
    return out


def e3_specs(backend: Any, arch: L.Architecture, *, repeats: int = REPEATS, n_random: int = N_RANDOM) -> list[CircuitSpec]:
    qubits = [backend.index_to_qubit_name(i) for i in range(backend.num_qubits)]
    p = preps(arch, qubits, n_random=n_random)
    out = [spec(f"E3_assign_{k}", p[k], 1, backend, arch, "assignment") for k in ["all0", "all1"] + [k for k in p if k.startswith("rand")]]
    out += [spec(f"E3_repeat_{k}_k{repeats:02d}", p[k], repeats, backend, arch, "repeated") for k in ("all0", "all1", "chkA", "chkB")]
    return out
