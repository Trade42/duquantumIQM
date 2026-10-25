"""The Garnet suite (E1–E5), in priority order, and the hardware job plan.

:func:`build` makes every circuit from one calibration: the placements are chosen from it
(``memory.rank``, ``repetition.best_path``, ``stability.best_block``), so a new calibration set can
move them; rebuild the suite from the snapshot taken just before a batch.

The job plan
------------
One job per circuit group, every circuit of a job at the same shot count (IQM's rule for a
batch). IQM runs a batch's circuits one after another, so drift inside a job would line up with
the circuit order: each job's order is a seeded random permutation, recorded per shot as
``position_in_job``. The pilot comes first:

* ``P0`` the pilot: E1, Z, R = 3, patch pA, 1,000 shots.

There is no barrier A/B (the old ``P1``). pulla-check compiled the pilot both ways with IQM's own
compiler: without barriers the readouts stop multiplexing and the circuit takes 7,748 ns against
3,900 ns with them (``docs/pulla-check.md``), so barriers stay on. The barrier-off circuit is still
built (:func:`build`) and its files stay in the record; no job runs it.

Then three batches, time-separated (at least 2 h apart, and across a calibration change if one
comes), each opened and closed by a calibration snapshot (E6, no QPU time). Every batch has the
readout characterisation (E3) and the two E1 patches; each adds one of E2, E4, E5, so the three
are spread evenly.

The ML split is fixed before any data exist (:data:`SPLIT`, :func:`split`): the whole of batch 3
is the **test** set and is never fine-tuned on; in every other job (the pilot phase included) a
seeded draw of shot indices, the same for every circuit of the job, makes 20% **validation** and
the rest **finetune**. Every shot's split is in ``shots.csv``, and the rule in each manifest.

QPU time is estimated per job as Σ shots × (circuit duration + per-shot overhead) + circuits ×
per-circuit overhead. Circuit durations come from :func:`dqiqm.experiments.circuits.schedule` with
the calibration's durations. The conservative per-shot overhead is the device's passive-reset wait
(``gates.reset_wait``, 298 µs today; any other per-shot cost, :data:`ELECTRONICS_PER_SHOT_S`, is
taken as 0); per circuit, :data:`PER_CIRCUIT_OVERHEAD_S`. The pilot's job timeline spans only
50.7 µs per shot from execution start to end, less than the reset wait alone; its raw shots show the
qubits did wait at least about 200 µs between shots (``orchestration/reports/pilots-review.md``), so
the span does not bracket the shot loop and is not a planning figure. Garnet's published CLOPS of
2,600 (IQM, arXiv:2408.12433) bounds the whole per-shot cost, loading included, at about 1.9 ms
(:data:`PESSIMISTIC_PER_SHOT_S`).
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dqiqm import layout as L
from dqiqm.experiments import device as dev
from dqiqm.experiments import memory, readout, repetition, stability
from dqiqm.experiments.circuits import CircuitSpec, schedule

#: Assumed per-shot cost beyond the reset wait (readout electronics, triggering), seconds: unknown
#: until the pilot's timeline, so 0 (the estimate is circuit duration + reset_wait per shot).
ELECTRONICS_PER_SHOT_S = 0.0
#: Per-shot overhead when the calibration serves no reset wait, seconds.
PER_SHOT_OVERHEAD_S = 1.0e-3
#: The CLOPS bound: 100 templates x 10 parameter sets x 100 shots x 5 layers / 2,600 CLOPS over 1e5 shots.
PESSIMISTIC_PER_SHOT_S = 100 * 10 * 100 * 5 / 2600 / 1e5
#: Assumed per-circuit overhead (loading a circuit of a batch), seconds.
PER_CIRCUIT_OVERHEAD_S = 0.1

SHOTS = {"P0": 1000, "E1": 5000, "E2": 5000, "E3": 2000, "E4": 5000, "E5": 5000}
BATCH_EXTRA = {1: "E2", 2: "E4", 3: "E5"}
#: The fine-tuning split: batch 3 held out whole as the test set; elsewhere 20% validation by shot index.
SPLIT = {"test_batch": 3, "validation_fraction": 0.2, "seed": 2026}


def split(job_label: str, batch: int, shots: int, rule: Mapping[str, Any] = SPLIT) -> np.ndarray:
    """Each shot's split: "test" for the held-out batch, else "validation" or "finetune" (seeded per job)."""
    if batch == rule["test_batch"]:
        return np.full(shots, "test", dtype=object)
    rng = np.random.default_rng([int(rule["seed"]), zlib.crc32(job_label.encode())])
    out = np.full(shots, "finetune", dtype=object)
    out[rng.permutation(shots)[: int(round(rule["validation_fraction"] * shots))]] = "validation"
    return out


def split_rule(job_label: str, batch: int, labels: np.ndarray, rule: Mapping[str, Any] = SPLIT) -> dict[str, Any]:
    vals, counts = np.unique(labels, return_counts=True)
    return {
        "rule": "batch test_batch is the test set, never fine-tuned on; in every other job, a seeded draw of "
                "round(validation_fraction * shots) shot indices (numpy default_rng([seed, crc32(job label)]).permutation) "
                "is validation, the rest finetune; the same indices for every circuit of the job",
        **dict(rule),
        "job": job_label,
        "role": "test" if batch == rule["test_batch"] else "finetune+validation",
        "counts": {str(v): int(c) for v, c in zip(vals, counts)},
    }


@dataclass
class Suite:
    cal: dev.Calibration
    backend: Any
    arch: L.Architecture
    durations: dev.Durations
    specs: list[CircuitSpec]
    choices: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._by_name = {s.name: s for s in self.specs}
        if len(self._by_name) != len(self.specs):
            raise ValueError("circuit names are not unique")
        for s in self.specs:
            if s.hardware.target != "hardware":
                raise ValueError(f"{s.name}: only target='hardware' circuits may be in the suite")

    def __getitem__(self, name: str) -> CircuitSpec:
        return self._by_name[name]

    def of(self, experiment: str) -> list[CircuitSpec]:
        return [s for s in self.specs if s.experiment == experiment]


def build(
    cal: dev.Calibration | None = None,
    *,
    experiments: Iterable[str] = ("E1", "E2", "E3", "E4", "E5"),
    e1_rounds: Sequence[int] = memory.E1_ROUNDS,
    e2_rounds: Sequence[int] = repetition.E2_ROUNDS,
    e4_rounds: Sequence[int] = memory.E4_ROUNDS,
    e4_delays_ns: Sequence[float] = memory.E4_DELAYS_NS,
    e5_rounds: Sequence[int] = stability.E5_ROUNDS,
    e3_random: int = readout.N_RANDOM,
    e3_repeats: int = readout.REPEATS,
    rank_shots: int = 50_000,
) -> Suite:
    """Every circuit of the suite on the stand-in Garnet with ``cal``'s noise (the live snapshot by default)."""
    cal = cal or dev.live_calibration()
    backend = dev.fake_backend(cal)
    arch = L.Architecture.from_backend(backend)
    dur = dev.Durations.from_calibration(cal)
    experiments = tuple(experiments)
    specs: list[CircuitSpec] = []
    choices: dict[str, Any] = {}

    ranked = memory.rank(arch, cal, shots=rank_shots)
    patches = memory.choose_patches(ranked)
    choices["E1"] = {
        "placements": [
            {"orientation": r["placement"].orientation, "origin": list(r["placement"].origin),
             "x_ancillas": list(r["x_ancillas"]), "p_l_r3": r["p_l"], "mean_p_l_r3": r["mean_p_l"],
             "patch": next((k for k, v in patches.items() if v is r), None)}
            for r in ranked
        ],
        "rank_shots": rank_shots,
        "qubits": sorted({q for _, q in ranked[0]["placement"].mapping}, key=dev.qubit_sort_key),
    }
    pl = {k: v["placement"] for k, v in patches.items()}
    if "E1" in experiments:
        specs += memory.e1_specs(pl, backend, e1_rounds)
        # the barrier-off pilot of the dropped P1 A/B: kept so the record's circuits stay as built; no job runs it
        specs.append(memory.spec(3, "Z", pl["pA"], "pA", backend, barriers=False))
    if "E3" in experiments:
        specs += readout.e3_specs(backend, arch, repeats=e3_repeats, n_random=e3_random)
    if "E2" in experiments:
        bp = repetition.best_path(arch, cal, dur)
        choices["E2"] = {k: (list(v) if k == "path" else v) for k, v in bp.items()}
        specs += repetition.e2_specs(bp["path"], backend, e2_rounds)
    if "E4" in experiments:
        specs += memory.e4_specs(pl["pA"], backend, rounds=e4_rounds, delays_ns=e4_delays_ns)
    if "E5" in experiments:
        bb = stability.best_block(arch, cal, dur)
        choices["E5"] = {k: v for k, v in bb.items() if k != "placement"} | {"qubits": dict(sorted(bb["placement"].physical.items()))}
        specs += stability.e5_specs(bb["placement"], backend, e5_rounds)
    return Suite(cal, backend, arch, dur, specs, choices)


# ---------------------------------------------------------------------------
# The job plan.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    label: str
    phase: str
    batch: int | None
    experiment: str
    circuits: tuple[str, ...]  # in run order (a seeded permutation)
    shots: int
    note: str = ""


def _order(names: Sequence[str], label: str) -> tuple[str, ...]:
    seed = int.from_bytes(label.encode(), "little") % (2**32)
    perm = np.random.default_rng(seed).permutation(len(names))
    return tuple(names[i] for i in perm)


def jobs(suite: Suite, shots: Mapping[str, int] = SHOTS, batches: int = 3) -> list[Job]:
    """The pilot, then ``batches`` batches (see the module docstring)."""
    e1 = [s.name for s in suite.of("E1") if s.params["barriers"]]
    pilot = next(s.name for s in suite.of("E1") if s.params["basis"] == "Z" and s.rounds == 3
                 and s.params["patch"] == "pA" and s.params["barriers"])
    out = [Job("P0", "pilot", None, "E1", (pilot,), shots["P0"], "sanity checks before anything else")]
    for b in range(1, batches + 1):
        groups = [("E3", [s.name for s in suite.of("E3")], "readout characterisation")]
        for patch in ("pA", "pB"):
            names = [n for n in e1 if suite[n].params["patch"] == patch]
            if names:
                groups.append(("E1", names, f"memory, patch {patch}"))
        extra = BATCH_EXTRA.get(b)
        if extra and suite.of(extra):
            groups.append((extra, [s.name for s in suite.of(extra)], {"E2": "repetition codes", "E4": "delay sweep",
                                                                       "E5": "stability"}[extra]))
        for i, (exp, names, note) in enumerate(groups, 1):
            label = f"B{b}-{i}-{exp}"
            out.append(Job(label, f"batch {b}", b, exp, _order(names, label), shots[exp], note))
    return out


def per_shot_overhead_s(cal: dev.Calibration) -> float:
    """The device's reset wait plus :data:`ELECTRONICS_PER_SHOT_S`, or :data:`PER_SHOT_OVERHEAD_S`."""
    if cal.reset_wait_ns:
        return cal.reset_wait_ns * 1e-9 + ELECTRONICS_PER_SHOT_S
    return PER_SHOT_OVERHEAD_S


def estimate(job: Job, suite: Suite, *, per_shot_s: float | None = None,
             per_circuit_s: float = PER_CIRCUIT_OVERHEAD_S) -> dict[str, Any]:
    per_shot_s = per_shot_overhead_s(suite.cal) if per_shot_s is None else per_shot_s
    durs = [schedule(suite[n].hardware, suite.durations).duration_ns for n in job.circuits]
    shots_s = sum(job.shots * (d * 1e-9 + per_shot_s) for d in durs)
    qpu = shots_s + per_circuit_s * len(durs)
    return {
        "label": job.label,
        "phase": job.phase,
        "batch": job.batch,
        "experiment": job.experiment,
        "circuits": len(job.circuits),
        "shots": job.shots,
        "total_shots": job.shots * len(job.circuits),
        "circuit_us_min": min(durs) / 1e3,
        "circuit_us_max": max(durs) / 1e3,
        "circuit_us_mean": float(np.mean(durs)) / 1e3,
        "qpu_s_shots": shots_s,
        "qpu_s_per_circuit_overhead": per_circuit_s * len(durs),
        "qpu_s": qpu,
        "qpu_s_pessimistic": sum(job.shots * (d * 1e-9 + PESSIMISTIC_PER_SHOT_S) for d in durs) + per_circuit_s * len(durs),
        "qpu_s_circuits_only": sum(job.shots * d * 1e-9 for d in durs),
        "per_shot_overhead_s": per_shot_s,
        "note": job.note,
        "order": list(job.circuits),
    }


def plan_json(suite: Suite, plan: Sequence[Mapping[str, Any]], *, pilot_span: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The plan as ``InfoRecord/circuits/job_plan.json`` holds it (``pilot_span``: the pilot's timeline span, for the record)."""
    return {
        "format": "dqiqm.inforecord/1",
        "calibration_set_id": suite.cal.calibration_set_id,
        "shots": SHOTS,
        "split": SPLIT,
        "per_shot_overhead_s": per_shot_overhead_s(suite.cal),
        "per_circuit_overhead_s": PER_CIRCUIT_OVERHEAD_S,
        "pessimistic_per_shot_s": PESSIMISTIC_PER_SHOT_S,
        **({"pilot_timeline_span": dict(pilot_span)} if pilot_span else {}),
        "choices": suite.choices,
        "jobs": list(plan),
    }
