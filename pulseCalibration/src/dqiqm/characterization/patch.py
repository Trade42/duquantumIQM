"""The pilot patch: its 17 qubits, its 24 couplers and the surface-code round's 4 parallel CZ layers.

Nothing here is typed in by hand. :func:`pilot_patch` rebuilds the pilot circuit
``E1_d3_Z_r03_pA`` with ``dqiqm.experiments.memory.spec`` on the committed placement
(``InfoRecord/circuits/E1/placement.json``), checks its ``native_sha256`` against the record, and
reads the CZ groups between barriers of its native circuit: 4 layers of 6 couplers, the same 4 in
every round, 24 distinct couplers in all, each a data-ancilla pair.

:func:`terminal_readout` reads the readout figures that apply to the characterization circuits.
Every measurement of C1 and C2 is terminal, and IQM's compiler turns a terminal ``measure`` into
``measure_fidelity`` (``docs/pulla-check.md``), so the priors are the quality-metric set's
**unheralded** ``metrics.ssro.measure_fidelity.constant.<q>.error_0_to_1`` / ``error_1_to_0``,
not ``measure``'s (``device.Calibration``) and not the heralded stash figures. QB20 is where it
matters: P(0|1) 0.090 for ``measure_fidelity`` against 0.017 for ``measure``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dqiqm import layout as L
from dqiqm.experiments import circuits as C
from dqiqm.experiments import device as dev

REPO_ROOT = dev.REPO_ROOT
PILOT = "E1_d3_Z_r03_pA"
PILOT_PLACEMENT = REPO_ROOT / "InfoRecord" / "circuits" / "E1" / "placement.json"
#: The live Garnet snapshot (resonance-check), calibration set a18a947f: the quality-metric set lives here.
SNAPSHOT = REPO_ROOT / "data" / "calibration" / "garnet" / "a18a947f-0408-4628-81ea-1715637e434f.20261024T191313Z.json"
#: pulla-check's compile file: the compiled readout windows per qubit.
COMPILE = REPO_ROOT / "data" / "calibration" / "garnet" / "pulse" / "compile.a18a947f-0408-4628-81ea-1715637e434f.json"
#: pulla-check's per-locus pulse parameters for the patch.
PULSE = REPO_ROOT / "data" / "calibration" / "garnet" / "pulse" / "a18a947f-0408-4628-81ea-1715637e434f.json"
DEVICE = "garnet"


@dataclass(frozen=True)
class Patch:
    """The pilot's qubits (sorted; logical index i of every characterization circuit is ``qubits[i]``),
    roles, couplers and CZ layers. A layer's pairs are (data, ancilla)."""

    qubits: tuple[str, ...]
    roles: Mapping[str, str]  # "data" or "ancilla"
    layers: tuple[tuple[tuple[str, str], ...], ...]
    pilot_native_sha256: str
    calibration_set_id: str

    @property
    def couplers(self) -> tuple[tuple[str, str], ...]:
        """The 24 couplers, as sorted pairs, sorted."""
        return tuple(sorted({dev.pair_key(*p) for layer in self.layers for p in layer}))

    def index(self, q: str) -> int:
        return self.qubits.index(q)

    def layer_of(self, a: str, b: str) -> int:
        """The 1-based CZ layer that holds coupler (a, b)."""
        key = dev.pair_key(a, b)
        for i, layer in enumerate(self.layers, 1):
            if key in {dev.pair_key(*p) for p in layer}:
                return i
        raise KeyError(f"{a}-{b} is not a coupler of the patch")

    def to_json(self) -> dict[str, Any]:
        return {
            "device": DEVICE,
            "source": f"{PILOT} rebuilt by dqiqm.experiments.memory.spec on InfoRecord/circuits/E1/placement.json",
            "pilot_native_sha256": self.pilot_native_sha256,
            "calibration_set_id": self.calibration_set_id,
            "qubits": list(self.qubits),
            "roles": dict(self.roles),
            "couplers": [list(p) for p in self.couplers],
            "layers": [[list(p) for p in layer] for layer in self.layers],
            "layer_note": "each pair is (data, ancilla); the same 4 layers run, in this order, in every round",
        }


def _placement(entry: Mapping[str, Any]) -> L.Placement:
    return L.Placement(
        tuple(entry["origin"]), int(entry["orientation"]), tuple(sorted((int(k), v) for k, v in entry["placement"].items()))
    )


def pilot_patch(backend: Any, cal: dev.Calibration | None = None) -> Patch:
    """The patch, read from the rebuilt pilot circuit; raises if the rebuild differs from the record."""
    from dqiqm.experiments import memory as M

    entry = json.loads(PILOT_PLACEMENT.read_text())[PILOT]
    spec = M.spec(3, "Z", _placement(entry), "pA", backend)
    sha = C.native_sha256(spec.hardware)
    if sha != entry["native_sha256"]:
        raise AssertionError(f"the rebuilt {PILOT} differs from the record: {sha} != {entry['native_sha256']}")
    qc = spec.hardware.logical
    names = {i: spec.physical[s] for i, s in enumerate(spec.hardware.stim_qubits)}
    roles = {spec.physical[s]: r for s, r in spec.roles.items()}
    groups: list[tuple[tuple[str, str], ...]] = []
    current: list[tuple[str, str]] = []
    for ci in qc.data:
        if ci.operation.name == "barrier":
            if current:
                groups.append(tuple(current))
                current = []
        elif ci.operation.name == "cz":
            a, b = (names[qc.find_bit(q).index] for q in ci.qubits)
            current.append((a, b) if roles[a] == "data" else (b, a))
    if current:
        groups.append(tuple(current))
    layers = tuple(tuple(sorted(g, key=lambda p: dev.qubit_sort_key(p[0]))) for g in groups[:4])
    rounds = int(entry["params"]["rounds"])
    if len(groups) != 4 * rounds or any(
        tuple(sorted(g, key=lambda p: dev.qubit_sort_key(p[0]))) != layers[i % 4] for i, g in enumerate(groups)
    ):
        raise AssertionError(f"{PILOT}: the CZ groups are not the same 4 layers in every round")
    patch = Patch(
        qubits=tuple(sorted(set(spec.physical.values()), key=dev.qubit_sort_key)),
        roles=dict(sorted(roles.items(), key=lambda kv: dev.qubit_sort_key(kv[0]))),
        layers=layers,
        pilot_native_sha256=sha,
        calibration_set_id=(cal.calibration_set_id if cal is not None else
                            json.loads((PILOT_PLACEMENT.parent.parent / "job_plan.json").read_text())["calibration_set_id"]),
    )
    _check(patch, L.Architecture.from_backend(backend))
    return patch


def _check(patch: Patch, arch: L.Architecture) -> None:
    if len(patch.qubits) != 17:
        raise AssertionError(f"{len(patch.qubits)} qubits, expected 17")
    if len(patch.couplers) != 24 or [len(layer) for layer in patch.layers] != [6, 6, 6, 6]:
        raise AssertionError("expected 24 couplers in 4 layers of 6")
    for layer in patch.layers:
        used = [q for p in layer for q in p]
        if len(set(used)) != len(used):
            raise AssertionError(f"a layer reuses a qubit: {layer}")
        for d, a in layer:
            if not arch.coupled(d, a):
                raise AssertionError(f"{d}-{a} is not a coupler of the device")
            if (patch.roles[d], patch.roles[a]) != ("data", "ancilla"):
                raise AssertionError(f"{d}-{a} is not a data-ancilla pair")


# ---------------------------------------------------------------------------
# Priors from the calibration: terminal readout, coherence, gate errors, durations.
# ---------------------------------------------------------------------------


def _observations(snapshot: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in ("calibration_set", "quality_metric_set"):
        for o in (snapshot.get(part) or {}).get("observations", []):
            if isinstance(o.get("value"), (int, float)):
                out[o["dut_field"]] = float(o["value"])
    return out


def terminal_readout(snapshot: Mapping[str, Any] | None = None, qubits: tuple[str, ...] | None = None) -> dict[str, dict[str, float]]:
    """Per qubit, the unheralded ``measure_fidelity`` assignment errors: ``p1_given_0``, ``p0_given_1``."""
    snapshot = snapshot if snapshot is not None else json.loads(SNAPSHOT.read_text())
    obs = _observations(snapshot)
    names = qubits or tuple(snapshot["architecture"]["qubits"])
    out = {}
    for q in names:
        base = f"metrics.ssro.measure_fidelity.constant.{q}"
        out[q] = {"p1_given_0": obs[f"{base}.error_0_to_1"], "p0_given_1": obs[f"{base}.error_1_to_0"]}
    return out


def readout_windows(path: Path = COMPILE) -> dict[str, dict[str, Any]]:
    """pulla-check's compiled readout windows: per qubit, ``probe_line``, ``measure_window_ns`` and
    ``measure_fidelity_window_ns`` (acquisition delay + integration + 76 ns, IQM's own arithmetic)."""
    return json.loads(Path(path).read_text())["readout_windows"]["qubits"]


@dataclass(frozen=True)
class Priors:
    """What the calibration says about each qubit and coupler, in the units the model uses.

    ``t1_ns``/``t2_ns`` (Ramsey T2, capped at 2 T1, as the stand-in uses), ``prx_ns``,
    ``error_1q`` (RB), ``p1_given_0``/``p0_given_1`` (terminal readout, unheralded
    ``measure_fidelity``); per coupler ``cz_error`` (IRB) and ``cz_ns``.
    """

    calibration_set_id: str
    qubits: Mapping[str, Mapping[str, float]]
    couplers: Mapping[tuple[str, str], Mapping[str, float]]
    terminal_window_ns: float  # the slowest terminal (measure_fidelity) readout window among the patch qubits

    def to_json(self) -> dict[str, Any]:
        return {
            "device": DEVICE,
            "calibration_set_id": self.calibration_set_id,
            "terminal_window_ns": self.terminal_window_ns,
            "qubits": {q: dict(v) for q, v in self.qubits.items()},
            "couplers": {f"{a}-{b}": dict(v) for (a, b), v in self.couplers.items()},
            "sources": {
                "t1_ns, t2_ns, prx_ns, error_1q, cz_error, cz_ns": "dqiqm.experiments.device.live_calibration() "
                "(T2 = Ramsey, capped at 2 T1)",
                "p1_given_0, p0_given_1": "metrics.ssro.measure_fidelity.constant.<q>.error_0_to_1 / error_1_to_0 "
                "(unheralded) of the snapshot's quality-metric set",
                "terminal_window_ns": "max over the patch of compile.*.json readout_windows measure_fidelity_window_ns",
            },
        }


def priors(patch: Patch, cal: dev.Calibration | None = None, snapshot: Mapping[str, Any] | None = None) -> Priors:
    cal = cal or dev.live_calibration()
    ro = terminal_readout(snapshot, patch.qubits)
    windows = readout_windows()
    qubits = {}
    for q in patch.qubits:
        c = cal.qubits[q]
        qubits[q] = {
            "t1_ns": float(c["t1_ns"]),
            "t2_ns": dev.t2_ns(c),
            "prx_ns": float(c["prx_ns"] or cal.median("prx_ns")),
            "error_1q": float(c["error_1q"]),
            **ro[q],
        }
    couplers = {p: {"cz_error": cal.cz_error(*p), "cz_ns": cal.cz_ns(*p)} for p in patch.couplers}
    window = max(float(windows[q]["measure_fidelity_window_ns"]) for q in patch.qubits)
    return Priors(cal.calibration_set_id, qubits, couplers, window)
