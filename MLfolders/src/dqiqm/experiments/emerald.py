"""The Garnet design on IQM Emerald, the challenge's device: E1 (and E4) on its two best patches, and the job plan.

Everything Garnet-specific in :mod:`.device`, :mod:`.memory` and :mod:`.suite` is wrapped here, not
edited (the emerald-design brief's "add, don't edit"):

The calibration
    The live snapshot of 2026-10-24 19:13 UTC (calibration set 23e12f6b), copied byte for byte into
    ``InfoRecord/emerald/calibration/emerald/`` (:data:`DEFAULT_SNAPSHOT`), read by
    :func:`dqiqm.experiments.device.calibration_from_snapshot` unchanged: readout from the default
    ``measure`` (``constant``) ``error_0_to_1`` / ``error_1_to_0``, not the heralded figure.

The couplers
    Emerald has 54 qubits and 90 static couplers; 5 of them have no CZ in this calibration set
    (:data:`CZ_LESS`). Every placement is searched on :func:`architecture`, the static grid with only
    the 85 calibrated couplers (``layout.Architecture.with_couplers``), and every circuit is checked
    against those 85 (:func:`check_live_couplers`), because ``layout.check_hardware`` and IQM's
    ``validate_circuit`` see the stand-in's 90.

The stand-in
    ``IQMFakeAphrodite`` has Emerald's 54 qubit names and 90 couplers, and
    ``copy_with_error_profile`` refuses a profile without all 90. :func:`error_profile` is
    ``device.error_profile`` on Aphrodite's loci, plus the 5 CZ-less couplers at a depolarizing
    parameter of 1.0 (:data:`CZ_LESS_SENTINEL`): no circuit uses them (checked), and one that did
    would decode at chance. Its T2 is capped at T1, not 2·T1 (:func:`t2_capped`: QB8, QB30, QB37,
    QB39 and QB50 today): with T2 > T1, Aer's thermal relaxation is a general Kraus channel, and
    Aer 0.17.2's matrix-product-state method fails on it ("QuantumError: Kraus is empty", or a
    segfault). That adds a little dephasing on 5 qubits, in the stand-in only. Like Garnet's
    stand-in, it puts no noise on ``delay`` or idle qubits.

Decoder priors
    :func:`calibrated_noise` is ``device.calibrated_noise(..., idle_ns=0)`` plus the idle of the
    readout window counted once: data qubits get the T1/T2 twirl's X/Y + Z part as
    ``idle_during_measurement``, ancillas its X/Y part, and nothing goes before round 1. This is
    the fix the garnet-design review asked for (its MAJOR 1), in a module of ours; with
    ``idle_ns=0`` it is ``device.calibrated_noise`` exactly.

Patches
    ``eA`` is the best placement under the calibrated Stim noise (``memory.rank``); ``eB`` the best
    placement on a *different* qubit set. Garnet has one 17-qubit set; Emerald has 15, so its
    second patch is a real spatial repeat, not a re-orientation.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dqiqm import layout as L
from dqiqm import surface as S
from dqiqm.experiments import device as dev
from dqiqm.experiments import memory
from dqiqm.experiments import suite as SU
from dqiqm.experiments.circuits import CircuitSpec, sent

DEVICE = "emerald"
REPO_ROOT = dev.REPO_ROOT
RECORD = REPO_ROOT / "InfoRecord" / "emerald"
CALSET = "23e12f6b-c23c-4736-8736-b0142989c0e6"
#: The resonance-check snapshot this design is built from (read-only, outside the record).
SNAPSHOT_SOURCE = REPO_ROOT / "data" / "calibration" / "emerald" / f"{CALSET}.20261024T191321Z.json"
#: Its byte copy inside the record, which the code reads.
DEFAULT_SNAPSHOT = RECORD / "calibration" / "emerald" / f"20261024T191321Z-{CALSET}.json"
#: The static couplers with no CZ in calibration set 23e12f6b (resonance-check, finding 4).
CZ_LESS = (("QB6", "QB7"), ("QB15", "QB23"), ("QB24", "QB32"), ("QB49", "QB50"), ("QB50", "QB54"))
#: The stand-in's depolarizing parameter on a CZ-less coupler: fully depolarizing, never used.
CZ_LESS_SENTINEL = 1.0
NATIVE_OPS = frozenset({"prx", "cz", "measure", "barrier", "delay"})

E1_ROUNDS = memory.E1_ROUNDS
E4_ROUNDS = memory.E4_ROUNDS
E4_DELAYS_NS = memory.E4_DELAYS_NS
PATCHES = ("eA", "eB")
#: Shots per circuit on hardware. Smaller jobs than Garnet's (11-12 circuits x 2,000 shots against
#: 22 x 5,000): Emerald is the challenge's device, so its queue is busier.
SHOTS = {"P0": 1000, "E1": 2000, "E4": 2000}
BATCHES = 3
#: Garnet's split, unchanged: batch 3 whole is the test set; 20% of shot indices elsewhere validation.
SPLIT = SU.SPLIT


# ---------------------------------------------------------------------------
# Calibration, couplers, the stand-in.
# ---------------------------------------------------------------------------


def live_calibration(path: str | Path = DEFAULT_SNAPSHOT) -> dev.Calibration:
    cal = dev.live_calibration(path)
    if cal.device != DEVICE:
        raise ValueError(f"{path}: a {cal.device!r} snapshot, not {DEVICE!r}")
    return cal


@functools.cache
def _aphrodite() -> Any:
    """One shared, noise-free ``IQMFakeAphrodite`` (1.7 s to build), read only: its names, loci and grid."""
    from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite

    return IQMFakeAphrodite()


def fake_loci(fake: Any = None) -> dict[frozenset, tuple[str, str]]:
    """Each of the stand-in's 90 couplers -> the ordered locus its architecture lists."""
    fake = fake or _aphrodite()
    return {frozenset(locus): tuple(locus) for info in fake.architecture.gates["cz"].implementations.values()
            for locus in info.loci}


def cz_less(cal: dev.Calibration, fake: Any = None) -> list[tuple[str, str]]:
    """The stand-in's couplers that ``cal`` has no CZ on."""
    live = {frozenset(k) for k in cal.couplers}
    return sorted((dev.pair_key(*k) for k in fake_loci(fake) if k not in live),
                  key=lambda p: (dev.qubit_sort_key(p[0]), dev.qubit_sort_key(p[1])))


def architecture(cal: dev.Calibration, fake: Any = None) -> L.Architecture:
    """Emerald's static grid with only the couplers ``cal`` calibrates (85 of 90 today)."""
    return L.Architecture.from_backend(fake or _aphrodite()).with_couplers(list(cal.couplers))


def error_profile(cal: dev.Calibration, *, t2: str = "ramsey") -> Any:
    """``device.error_profile`` on Aphrodite's loci, with the CZ-less couplers at :data:`CZ_LESS_SENTINEL`."""
    import dataclasses

    fake = _aphrodite()
    loci = fake_loci(fake)
    ep = dev.error_profile(cal, t2=t2, ordered_loci=loci)
    cz = dict(ep.two_qubit_gate_depolarizing_error_parameters["cz"])
    for pair in cz_less(cal, fake):
        cz[loci[frozenset(pair)]] = CZ_LESS_SENTINEL
    return dataclasses.replace(ep, t2s={q: min(v, ep.t1s[q]) for q, v in ep.t2s.items()},
                               two_qubit_gate_depolarizing_error_parameters={"cz": cz},
                               name=f"{DEVICE} {cal.calibration_set_id}")


def t2_capped(cal: dev.Calibration, *, t2: str = "ramsey") -> list[str]:
    """The qubits whose stand-in T2 :func:`error_profile` caps at T1 (their T2 is above T1)."""
    return sorted((q for q, c in cal.qubits.items() if dev.t2_ns(c, t2) > float(c["t1_ns"])), key=dev.qubit_sort_key)


def fake_backend(cal: dev.Calibration | None = None, *, t2: str = "ramsey") -> Any:
    """``IQMFakeAphrodite()``, with ``cal``'s noise when given."""
    fake = _aphrodite()
    return fake if cal is None else fake.copy_with_error_profile(error_profile(cal, t2=t2))


def stand_in_name(cal: dev.Calibration) -> str:
    return f"IQMFakeAphrodite+profile({cal.calibration_set_id[:8]})"


def check_live_couplers(spec: CircuitSpec, cal: dev.Calibration) -> dict[str, int]:
    """Raise ``AssertionError`` unless every sent op is native and every ``cz`` is on a coupler ``cal`` calibrates."""
    live = {frozenset(k) for k in cal.couplers}
    dead = {frozenset(k) for k in CZ_LESS}
    n_cz = 0
    for inst in sent(spec.hardware):
        if inst.name not in NATIVE_OPS:
            raise AssertionError(f"{spec.name}: non-native op {inst.name!r} on {inst.locus}")
        if inst.name == "cz":
            pair = frozenset(inst.locus)
            if pair not in live or pair in dead:
                raise AssertionError(f"{spec.name}: cz on {inst.locus}, not a calibrated coupler")
            n_cz += 1
    if "swap" in spec.hardware.circuit.count_ops():
        raise AssertionError(f"{spec.name}: a SWAP was inserted")
    return {"cz": n_cz, "swaps": 0}


# ---------------------------------------------------------------------------
# Decoder priors with the readout-window idle counted once.
# ---------------------------------------------------------------------------


def calibrated_noise(cal: dev.Calibration, spec: CircuitSpec, *, idle_ns: float = 0.0) -> S.NoiseModel:
    """``device.calibrated_noise(..., idle_ns=0)`` plus, when ``idle_ns > 0``, one idle per readout window.

    Data qubits: ``idle_during_measurement`` = the twirl's p_xy + p_z over ``idle_ns`` (it is applied
    after each ancilla readout; ``surface.idle_data`` also fires after the last round's, which is
    conservative). Ancillas: p_xy (a Z does nothing to a measured ancilla). Nothing before round 1.
    """
    nm = dev.calibrated_noise(cal, spec.physical, spec.roles, pairs=spec.cx_pairs, idle_ns=0.0,
                              x_ancillas=spec.x_ancillas)
    if idle_ns <= 0:
        return nm
    rates = {}
    for s, r in nm.qubit_rates.items():
        c = cal.qubits[spec.physical[s]]
        pxy, pz = dev.relaxation_pauli(idle_ns, float(c["t1_ns"]), dev.t2_ns(c))
        extra = min(pxy + pz, 0.75) if spec.roles.get(s, "data") == "data" else min(pxy, 0.5)
        rates[s] = {**r, "idle_during_measurement": extra}
    return S.NoiseModel(qubit_rates=rates, pair_rates=dict(nm.pair_rates))


def readout_window_ns(spec: CircuitSpec, durations: dev.Durations) -> float:
    """The ancilla readout layer's length: its slowest ancilla's ``measure`` duration."""
    return max(durations.measure_ns[spec.physical[a]] for a in spec.stabilizers)


# ---------------------------------------------------------------------------
# Placements and the suite.
# ---------------------------------------------------------------------------


def qubit_set(r: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(r["placement"].physical.values())


def choose_patches(ranked: Sequence[dict]) -> dict[str, dict]:
    """``eA``: the best placement; ``eB``: the best placement on another qubit set."""
    best = ranked[0]
    other = next((r for r in ranked if qubit_set(r) != qubit_set(best)), None)
    return {"eA": best} | ({"eB": other} if other is not None else {})


def ranking_by_set(ranked: Sequence[dict]) -> list[dict]:
    """One row per qubit set, its best orientation, best first."""
    seen: dict[frozenset, dict] = {}
    for r in ranked:
        seen.setdefault(qubit_set(r), r)
    return list(seen.values())


def build(
    cal: dev.Calibration | None = None,
    *,
    experiments: Iterable[str] = ("E1", "E4"),
    e1_rounds: Sequence[int] = E1_ROUNDS,
    e4_rounds: Sequence[int] = E4_ROUNDS,
    e4_delays_ns: Sequence[float] = E4_DELAYS_NS,
    rank_shots: int = 50_000,
) -> SU.Suite:
    """E1 on eA and eB (and E4 on eA) on the stand-in Emerald, every circuit on the 85 live couplers."""
    cal = cal or live_calibration()
    backend = fake_backend(cal)
    arch = architecture(cal)
    dur = dev.Durations.from_calibration(cal)
    experiments = tuple(experiments)
    ranked = memory.rank(arch, cal, shots=rank_shots)
    patches = choose_patches(ranked)
    by_set = ranking_by_set(ranked)
    choices: dict[str, Any] = {"E1": {
        "device": DEVICE,
        "cz_less_couplers_excluded": [list(p) for p in CZ_LESS],
        "rule": "eA: the lowest mean P_L(R=3, Z and X) under calibrated Stim noise over every zero-SWAP placement on the "
                "calibrated couplers; eB: the best placement on a different qubit set",
        "rank_shots": rank_shots,
        "placements": [
            {"orientation": r["placement"].orientation, "origin": list(r["placement"].origin),
             "qubit_set": by_set.index(next(x for x in by_set if qubit_set(x) == qubit_set(r))),
             "x_ancillas": list(r["x_ancillas"]), "p_l_r3": r["p_l"], "mean_p_l_r3": r["mean_p_l"],
             "patch": next((k for k, v in patches.items() if v is r), None)}
            for r in ranked
        ],
        "qubit_sets": [sorted(qubit_set(r), key=dev.qubit_sort_key) for r in by_set],
        "patches": {k: sorted(qubit_set(v), key=dev.qubit_sort_key) for k, v in patches.items()},
    }}
    pl = {k: v["placement"] for k, v in patches.items()}
    specs: list[CircuitSpec] = []
    if "E1" in experiments:
        specs += memory.e1_specs(pl, backend, e1_rounds)
    if "E4" in experiments:
        specs += memory.e4_specs(pl["eA"], backend, patch="eA", rounds=e4_rounds, delays_ns=e4_delays_ns)
    for s in specs:
        s.checks["live_couplers"] = check_live_couplers(s, cal)  # type: ignore[index]
    return SU.Suite(cal, backend, arch, dur, specs, choices)


# ---------------------------------------------------------------------------
# The job plan.
# ---------------------------------------------------------------------------


def pilot_name(suite: SU.Suite) -> str:
    return next(s.name for s in suite.of("E1") if s.params["basis"] == "Z" and s.rounds == 3 and s.params["patch"] == "eA")


def jobs(suite: SU.Suite, shots: Mapping[str, int] = SHOTS, batches: int = BATCHES) -> list[SU.Job]:
    """The pilot, then ``batches`` batches of E1 (eA Z, eA X, eB Z, eB X: one job each), E4 in batch 2.

    Labels start with ``E`` (``EP0``, ``E-B1-1-E1``), so they never collide with Garnet's in the
    ledger or in the split's seeds. There is no barrier A/B on Emerald: Garnet's P1 decides it.
    """
    out = [SU.Job("EP0", "pilot", None, "E1", (pilot_name(suite),), shots["P0"], "sanity checks before anything else")]
    for b in range(1, batches + 1):
        groups = []
        for patch in PATCHES:
            for basis in memory.BASES:
                names = [s.name for s in suite.of("E1") if s.params["patch"] == patch and s.params["basis"] == basis]
                if names:
                    groups.append(("E1", names, f"memory, patch {patch}, {basis} basis"))
        if b == 2:
            for basis in memory.BASES:
                names = [s.name for s in suite.of("E4") if s.params["basis"] == basis]
                if names:
                    groups.append(("E4", names, f"delay sweep, patch eA, {basis} basis"))
        for i, (exp, names, note) in enumerate(groups, 1):
            label = f"E-B{b}-{i}-{exp}"
            out.append(SU.Job(label, f"batch {b}", b, exp, SU._order(names, label), shots[exp], note))  # noqa: SLF001
    return out


def plan_json(suite: SU.Suite, plan: Sequence[Mapping[str, Any]], pilot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The plan as ``InfoRecord/emerald/circuits/job_plan.json`` holds it."""
    return {
        "format": "dqiqm.inforecord/1",
        "device": DEVICE,
        "calibration_set_id": suite.cal.calibration_set_id,
        "shots": dict(SHOTS),
        "split": dict(SPLIT),
        "per_shot_overhead_s": SU.per_shot_overhead_s(suite.cal),
        "per_circuit_overhead_s": SU.PER_CIRCUIT_OVERHEAD_S,
        "pessimistic_per_shot_s": SU.PESSIMISTIC_PER_SHOT_S,
        "pessimistic_note": "Garnet's CLOPS bound (2,600 CLOPS), used for Emerald too: no Emerald CLOPS figure was checked",
        "choices": suite.choices,
        "jobs": list(plan),
        "pilot": dict(pilot) if pilot is not None else None,
    }


# ---------------------------------------------------------------------------
# d = 5 feasibility.
# ---------------------------------------------------------------------------


def d5_feasibility(cal: dev.Calibration) -> dict[str, Any]:
    """Zero-SWAP d=5 placements on the calibrated couplers, and on all 90 (with the CZ-less ones it needs)."""
    fake = _aphrodite()
    live = architecture(cal, fake)
    full = L.Architecture.from_backend(fake).with_couplers([tuple(k) for k in fake_loci(fake)])
    source = S.memory_circuit(5, 2, "Z")
    g = S.geometry(5)
    pairs = [p for layer in g.cx_layers for p in layer]
    out: dict[str, Any] = {"live_placements": len(L.placements(source, live))}
    full_p = L.placements(source, full)
    sets = {}
    for p in full_p:
        used = {frozenset((p.physical[a], p.physical[b])) for a, b in pairs}
        needs = sorted((dev.pair_key(*k) for k in used if k in {frozenset(c) for c in CZ_LESS}),
                       key=lambda q: (dev.qubit_sort_key(q[0]), dev.qubit_sort_key(q[1])))
        key = frozenset(p.physical.values())
        sets.setdefault(key, []).append({"orientation": p.orientation, "needs_cz_less": [list(x) for x in needs]})
    out["all_90_placements"] = len(full_p)
    out["all_90_qubit_sets"] = [{"qubits": sorted(k, key=dev.qubit_sort_key), "orientations": v} for k, v in sets.items()]
    return out
