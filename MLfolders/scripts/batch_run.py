"""batches: the Garnet and Emerald batches. 6 segments, 26 jobs, 1,419,000 shots, one job at a time.

    uv run python scripts/batch_run.py plan                                     # offline: the brief's table against both job plans
    uv run python scripts/batch_run.py snapshot --segment S                     # live, read-only: resonance_check.py + live checks
    uv run python scripts/batch_run.py verify --segment S [--snapshot P [--live] [--rerank]]
    DQ_QPU=run DQ_QPU_TASK=S DQ_QPU_MAX_JOBS=<jobs> DQ_QPU_MAX_SHOTS=<shots> \\
        uv run python scripts/batch_run.py submit --segment S --snapshot P      # the segment's jobs, in plan order
    uv run python scripts/batch_run.py recover --segment S --job-id J           # read-only; only if submit failed after a job id
    uv run python scripts/batch_run.py analyze --segment S --snapshots P Q ...  # offline: the records (run under the sim lock)
    uv run python scripts/batch_run.py e3 --segment garnet-b1                   # offline: E3's readout features, pooled per qubit
    uv run python scripts/batch_run.py summary                                  # offline: pooled LER fits, X against Z, docs/batches.md

The segments are the brief's QPU allowance (``SEGMENTS``), in this order: ``garnet-b1``, ``garnet-b2``,
``emerald-b1``, ``emerald-b2``, ``garnet-b3``, ``emerald-b3``. Each is armed on its own task name with
exactly its row's jobs and shots. Only the labels in the table can be submitted: the pilots (P0, EP0)
and Garnet's dropped barrier A/B (P1) are in no segment.

This generalises ``scripts/garnet_pilot.py`` and ``scripts/emerald_pilot.py`` (snapshot, verify, pin,
submit, recover, analyze) from one circuit to a segment of multi-circuit jobs. Those scripts are
unchanged; what is copied from them is said where it is.

The freeze (emerald-design review, MINOR 1, for both devices)
    Every circuit is rebuilt from the **committed** snapshot the plan was made from
    (``suite.build(device.live_calibration(device.DEFAULT_SNAPSHOT))`` for Garnet, calibration set
    a18a947f; ``emerald.build(emerald.live_calibration(emerald.DEFAULT_SNAPSHOT))`` for Emerald, 23e12f6b)
    and its ``native_sha256`` must equal its ``placement.json`` entry: a mismatch stops everything. The
    job plan's circuit order must equal the rebuilt ``jobs()``'s. A live calibration never rebuilds a
    circuit. Against a live snapshot, every circuit's own qubits must still serve ``prx`` and ``measure``,
    and every ``cz`` locus it uses must still be calibrated (Emerald: ``emerald.check_live_couplers`` too);
    IQM's ``create_run_request`` then validates the request against the live architecture. A frozen
    circuit that lost a qubit or coupler stops the segment; a changed calibration set otherwise does not.
    ``verify --rerank`` records the re-rank under a changed set, for information only.
    ``circuits/`` is never regenerated.

The qubits (pinned as the pilots established from iqm-client 35.0.3's code)
    One job is one ``hw.submit`` of the job's circuits in ``job_plan.json`` order with one
    ``qubit_index_to_name``: the union of the circuits' own maps (each is the stand-in's
    ``index_to_qubit_name`` on that circuit's qubits, so they never conflict; checked).
    ``create_run_request`` only serializes; each circuit's instructions are named through that map, so
    the live backend's index order never enters. Before each job, the very request ``hw.submit`` will
    build is built on the read-only backend and checked circuit by circuit: its instructions hash to the
    circuit's planned ``native_sha256``, ``hw._sent_qubits`` is that circuit's placement qubits,
    ``qubit_mapping`` is None, heralding none, DD disabled, ``active_reset_cycles=None``.

Submitting
    ``submit`` refuses unless armed exactly as the segment's row, against the default ledger, with every
    earlier segment complete in the ledger. Before each job it re-reads the server's default
    calibration set: if it moved, it takes a new snapshot (``resonance_check.py``), runs the live checks
    on it, and continues on the new set (recorded). It writes the job's manifest
    (``raw/<YYYYMMDD>-<task>/<label>.json``: position -> circuit, ``native_sha256``, qubits) **before**
    ``hw.submit``, prints the job id the moment ``hw.submit`` writes the ledger row, then checks the row,
    ``meta.json`` and every circuit's sanity checks. A failed check stops the segment. A job already in
    the ledger (its purpose names the task and label) is never submitted again; a job whose raw data is
    missing is recovered read-only (``recover``: ``hw.retrieve`` + ``hw.save_raw``), never resubmitted.

The sanity checks, per circuit (the brief's list)
    the measurement shape; all-zero rows; the first readout of every non-data qubit (E1 and E4 Z: the
    X-type ancillas random and the Z-type low; X bases the reverse; E2, E3 and E5 their own pattern), read
    from a noiseless sample of the circuit's Stim source rather than hard-coded columns; the detection
    fraction well below 0.5; ``meta.json``'s ``physical_qubits`` equal to the plan's. The decoder
    priors' own detection fraction (``hardware_noise``: the readout window plus E4's delay) is reported
    beside it as information, not as a stop.

The records
    ``analyze`` writes every run with the record's own writers (``datasets/``, ``derived/``, ``ler/``),
    labelled HARDWARE, with ``device``, ``batch`` and the split (B1, B2 finetune and validation; B3 test).
    The decoder is PyMatching on the noisy copy of the circuit that ran, with the corrected hardware
    priors, ``device.hardware_noise(cal, spec)`` (garnet-timing, main ``79092cf``; the orchestrator's
    instruction replaces the brief's pilot priors): the symmetric idle over the compiled readout window
    plus E4's delay, ``measure_fidelity`` errors at the final readout, no idle after the last round of a
    surface memory. ``cal`` is the snapshot of the job's own calibration set. Emerald runs the same
    function: no compile file exists for Emerald, so its window falls back on ``measure_ns`` (360 ns,
    the probe alone), as ``hardware_noise`` logs and the record states. Every dataset's decoder note,
    LER record and summary says ``priors: hardware_noise (79092cf)``. Nothing is chosen on test.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dqiqm import decode as D
from dqiqm import hw
from dqiqm import inforecord as IR
from dqiqm import stats
from dqiqm.experiments import circuits as C
from dqiqm.experiments import device as dev
from dqiqm.experiments import emerald as EM
from dqiqm.experiments import suite as SU

REPO = Path(__file__).resolve().parents[1]
CALIBRATION = REPO / "data" / "calibration"
DOC = REPO / "docs" / "batches.md"
PROGRESS = Path("/home/sal/duquantumIQM/orchestration/reports/batches-progress.md")

NATIVE_OPS = frozenset({"prx", "cz", "measure", "barrier", "delay"})
#: Noiseless samples that tell a deterministic measurement from a random one, and the model's shots.
REF_SHOTS, MODEL_SHOTS, SEED = 1024, 20_000, 2026
#: The sanity bands. A random first readout pools to [0.4, 0.6] (the pilots' band); a deterministic
#: one misreads under 0.3 (the pilots' Z-type band); the detection fraction stays well below 0.5.
RANDOM_BAND = (0.4, 0.6)
DETERMINISTIC_MAX = 0.3
DETECTION_MAX = 0.45
#: Where the calibrated model itself predicts more than DETECTION_HIGH_MODEL (E4 at 5 us: about 0.4), the
#: limit is DETECTION_MAX_HIGH: still below a dead or scrambled record's 0.5.
DETECTION_HIGH_MODEL, DETECTION_MAX_HIGH = 0.3, 0.48
#: All-zero rows: at most 5 per 1,000 (the pilots' band), or twice what the noiseless and model records give.
ALL_ZERO_PER_1000 = 5


def log(msg: str = "") -> None:
    print(msg, flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def _nan_to_none(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _nan_to_none(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_nan_to_none(x) for x in v]
    if isinstance(v, np.generic):
        return _nan_to_none(v.item())
    return v


def write_json(path: Path, body: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_nan_to_none(body), indent=1, default=str) + "\n")
    tmp.replace(path)


def tag_json(path: Path, **keys: Any) -> None:
    """Add ``keys`` at the top of a JSON object file (NaN written as null): ``scripts/emerald_pilot.py``'s."""
    body = json.loads(path.read_text())
    path.write_text(json.dumps(_nan_to_none({**keys, **body}), indent=1) + "\n")


# ---------------------------------------------------------------------------
# The allowance: the brief's table, and the two devices.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    task: str
    device: str
    batch: int
    labels: tuple[str, ...]
    jobs: int
    shots: int

    @property
    def split(self) -> str:
        return "test" if self.batch == SU.SPLIT["test_batch"] else "finetune+validation"

    def arming(self) -> dict[str, str]:
        return {"DQ_QPU": "run", "DQ_QPU_TASK": self.task, "DQ_QPU_MAX_JOBS": str(self.jobs),
                "DQ_QPU_MAX_SHOTS": str(self.shots)}


#: The brief's QPU allowance, per segment, in its exact order.
SEGMENTS: dict[str, Segment] = {s.task: s for s in (
    Segment("garnet-b1", "garnet", 1, ("B1-1-E3", "B1-2-E1", "B1-3-E1", "B1-4-E2"), 4, 399_000),
    Segment("garnet-b2", "garnet", 2, ("B2-1-E3", "B2-2-E1", "B2-3-E1", "B2-4-E4"), 4, 384_000),
    Segment("emerald-b1", "emerald", 1, ("E-B1-1-E1", "E-B1-2-E1", "E-B1-3-E1", "E-B1-4-E1"), 4, 88_000),
    Segment("emerald-b2", "emerald", 2, ("E-B2-1-E1", "E-B2-2-E1", "E-B2-3-E1", "E-B2-4-E1", "E-B2-5-E4", "E-B2-6-E4"),
            6, 136_000),
    Segment("garnet-b3", "garnet", 3, ("B3-1-E3", "B3-2-E1", "B3-3-E1", "B3-4-E5"), 4, 324_000),
    Segment("emerald-b3", "emerald", 3, ("E-B3-1-E1", "E-B3-2-E1", "E-B3-3-E1", "E-B3-4-E1"), 4, 88_000),
)}
ORDER = tuple(SEGMENTS)
TOTAL_JOBS, TOTAL_SHOTS = 26, 1_419_000


@dataclass(frozen=True)
class DeviceConf:
    name: str
    record: Path
    design_calset: str
    committed_snapshot: Path

    @property
    def raw_root(self) -> Path:
        return self.record / "raw"

    @property
    def circuits(self) -> Path:
        return self.record / "circuits"

    @property
    def job_plan(self) -> Path:
        return self.circuits / "job_plan.json"


DEVICES = {
    "garnet": DeviceConf("garnet", REPO / "InfoRecord", "a18a947f-0408-4628-81ea-1715637e434f", dev.DEFAULT_SNAPSHOT),
    "emerald": DeviceConf("emerald", EM.RECORD, EM.CALSET, EM.DEFAULT_SNAPSHOT),
}


def live_calibration(device: str, path: Path) -> dev.Calibration:
    cal = EM.live_calibration(path) if device == "emerald" else dev.live_calibration(path)
    if cal.device != device:
        raise SystemExit(f"{path}: a {cal.device!r} calibration, not {device!r}")
    return cal


def load_snapshot(device: str, path: Path) -> dict[str, Any]:
    snap = json.loads(Path(path).read_text())
    if snap.get("device") != device:
        raise SystemExit(f"{path}: device {snap.get('device')!r}, not {device}")
    return snap


@dataclass(frozen=True)
class PlannedJob:
    label: str
    experiment: str
    batch: int
    names: tuple[str, ...]
    shots: int
    position: int  # in the segment, 0-based

    @property
    def total_shots(self) -> int:
        return self.shots * len(self.names)


def check_plan(seg: Segment, plan: Mapping[str, Any], built: Mapping[str, Any] | None = None) -> list[PlannedJob]:
    """The segment's jobs from ``job_plan.json``, checked against the brief's table (and the rebuilt ``jobs()``).

    Raises ``SystemExit`` (stop and report) on any disagreement.
    """
    if plan.get("device", "garnet") != seg.device:
        raise SystemExit(f"{seg.task}: the job plan is {plan.get('device', 'garnet')!r}'s, not {seg.device!r}'s")
    entries = {j["label"]: j for j in plan["jobs"]}
    out = []
    for i, label in enumerate(seg.labels):
        j = entries.get(label)
        if j is None:
            raise SystemExit(f"{seg.task}: {label} is not in the job plan")
        order = tuple(j["order"])
        problems = []
        if j["batch"] != seg.batch:
            problems.append(f"batch {j['batch']} != {seg.batch}")
        if len(order) != j["circuits"] or len(set(order)) != len(order):
            problems.append(f"order has {len(order)} circuits ({len(set(order))} distinct), the plan says {j['circuits']}")
        if j["total_shots"] != j["shots"] * j["circuits"]:
            problems.append(f"total_shots {j['total_shots']} != {j['shots']} x {j['circuits']}")
        if not label.endswith(j["experiment"]):
            problems.append(f"experiment {j['experiment']}")
        if built is not None:
            b = built.get(label)
            if b is None or tuple(b.circuits) != order or b.shots != j["shots"] or b.batch != seg.batch:
                problems.append("the rebuilt jobs() disagree with job_plan.json's order or shots")
        if problems:
            raise SystemExit(f"{seg.task}: {label}: " + "; ".join(problems))
        out.append(PlannedJob(label, j["experiment"], j["batch"], order, int(j["shots"]), i))
    jobs, shots = len(out), sum(j.total_shots for j in out)
    if (jobs, shots) != (seg.jobs, seg.shots):
        raise SystemExit(f"{seg.task}: the plan gives {jobs} jobs and {shots:,} shots; the brief allows {seg.jobs} and {seg.shots:,}")
    return out


def plan_json(device: str) -> dict[str, Any]:
    return json.loads(DEVICES[device].job_plan.read_text())


# ---------------------------------------------------------------------------
# The frozen circuits.
# ---------------------------------------------------------------------------


class Frozen:
    """A device's circuits rebuilt from the committed snapshot the plan was made from, never from a live one."""

    def __init__(self, device: str) -> None:
        self.conf = DEVICES[device]
        self.device = device
        self.cal = live_calibration(device, self.conf.committed_snapshot)
        if self.cal.calibration_set_id != self.conf.design_calset:
            raise SystemExit(f"FREEZE: the committed snapshot is {self.cal.calibration_set_id}, not {self.conf.design_calset}")
        self.suite = EM.build(self.cal) if device == "emerald" else SU.build(self.cal)
        self.built_jobs = {j.label: j for j in (EM.jobs(self.suite) if device == "emerald" else SU.jobs(self.suite))}
        self.plan = plan_json(device)
        if self.plan["calibration_set_id"] != self.conf.design_calset:
            raise SystemExit(f"{device}: job_plan.json is for {self.plan['calibration_set_id']}")
        self.placement: dict[str, dict[str, Any]] = {}
        for p in sorted(self.conf.circuits.glob("*/placement.json")):
            self.placement.update(json.loads(p.read_text()))

    def jobs(self, seg: Segment) -> list[PlannedJob]:
        return check_plan(seg, self.plan, self.built_jobs)

    def planned_qubits(self, name: str) -> list[str]:
        return sorted(set(self.placement[name]["placement"].values()), key=dev.qubit_sort_key)

    def names_map(self, job: PlannedJob) -> dict[int, str]:
        return merged_names([self.suite[n].hardware.qubit_index_to_name for n in job.names])

    def mappings(self, job: PlannedJob) -> list[dict[int, str]]:
        """What ``hw.submit`` writes to ``meta.json``'s ``qubit_mapping`` for this job."""
        return hw._qubit_mappings(None, [self.suite[n].hardware.circuit for n in job.names], self.names_map(job))  # noqa: SLF001


def merged_names(maps: Sequence[Mapping[int, str]]) -> dict[int, str]:
    """The union of the circuits' ``qubit_index_to_name`` maps. Raises ``SystemExit`` on a conflict."""
    out: dict[int, str] = {}
    for m in maps:
        for k, v in m.items():
            if out.setdefault(int(k), v) != v:
                raise SystemExit(f"qubit index {k} is {out[int(k)]} in one circuit and {v} in another")
    if len(set(out.values())) != len(out):
        raise SystemExit("two qubit indices map to one device qubit")
    return dict(sorted(out.items()))


def instructions_json(instructions: Sequence[Any]) -> list[dict[str, Any]]:
    """``circuits.native_json`` for any sequence of serialized instructions: ``scripts/garnet_pilot.py``'s."""
    return [{"name": i.name, "locus": list(i.locus),
             "args": {k: (float(v) if isinstance(v, float) else v) for k, v in i.args.items()}} for i in instructions]


def instructions_sha256(instructions: Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps(instructions_json(instructions), sort_keys=True).encode()).hexdigest()


def freeze_checks(fz: Frozen, job: PlannedJob) -> list[dict[str, Any]]:
    """Every circuit of a job, rebuilt from the committed snapshot, against ``placement.json``. Raises ``AssertionError``."""
    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey

    arch_couplers = {frozenset(c) for c in fz.suite.arch.couplers}
    out = []
    for name in job.names:
        spec, entry = fz.suite[name], fz.placement[name]
        hc = spec.hardware
        assert spec.experiment == job.experiment, (name, spec.experiment)
        assert hc.target == "hardware" and entry["target"] == "hardware", name
        native = C.native_sha256(hc)
        assert native == entry["native_sha256"], f"FREEZE: {name}: native_sha256 {native} != placement.json {entry['native_sha256']}"
        assert spec.stim_sha256() == entry["stim_sha256"], f"FREEZE: {name}: stim_sha256 differs from placement.json"
        qasm_sha = hashlib.sha256((C.qasm(hc) + "\n").encode()).hexdigest()
        assert qasm_sha == entry["qasm_sha256"], f"FREEZE: {name}: qasm_sha256 differs from placement.json"
        assert "swap" not in hc.circuit.count_ops(), f"{name}: a SWAP"
        sent = C.sent(hc)
        ops = Counter(i.name for i in sent)
        assert set(ops) <= NATIVE_OPS, f"{name}: non-native ops {set(ops) - NATIVE_OPS}"
        cz = [frozenset(i.locus) for i in sent if i.name == "cz"]
        assert all(c in arch_couplers for c in cz), f"{name}: a cz off the device's couplers"
        if fz.device == "emerald":
            EM.check_live_couplers(spec, fz.cal)
        qubits = sorted({q for i in sent for q in i.locus}, key=dev.qubit_sort_key)
        assert qubits == fz.planned_qubits(name), f"{name}: sent to {qubits}, placed on {fz.planned_qubits(name)}"
        n = spec.stim.num_measurements
        meas_q = C.measured_qubits(spec.stim)
        meas = [i for i in sent if i.name == "measure"]
        clbits = []
        for inst in meas:
            mk = MeasurementKey.from_string(inst.args["key"])
            assert mk.creg_idx == 0 and mk.creg_len == n, (name, mk)
            (qb,) = inst.locus
            assert qb == spec.physical[meas_q[mk.clbit_idx]], f"{name}: clbit {mk.clbit_idx} measured on {qb}"
            clbits.append(mk.clbit_idx)
        assert sorted(clbits) == list(range(n)), f"{name}: the measurements are not clbits 0..{n - 1}"
        (mapping,) = hw._qubit_mappings(None, [hc.circuit], hc.qubit_index_to_name)  # noqa: SLF001
        assert mapping == {v: spec.physical[s] for v, s in enumerate(hc.stim_qubits)}, f"{name}: qubit_mapping != the placement"
        out.append({"circuit": name, "native_sha256": native, "ops": dict(ops), "cz": len(cz), "couplers": len(set(cz)),
                    "measurements": n, "swaps": 0, "qubits": qubits})
    return out


def _served(arch: Mapping[str, Any], gate: str, locus: tuple[str, ...]) -> bool:
    """Whether the implementation that runs ``gate`` on ``locus`` lists it: ``scripts/garnet_pilot.py``'s."""
    info = arch["gates"].get(gate)
    if not info:
        return False
    impl = dev._impl(info, locus)  # noqa: SLF001 -- the rule calibration_from_snapshot reads the figures by
    loci = info["implementations"].get(impl, {}).get("loci", [])
    return any(len(lc) == len(locus) and set(lc) == set(locus) for lc in loci)


def circuit_gates(spec: Any) -> tuple[list[str], list[tuple[str, str]]]:
    """The device qubits a circuit's sent instructions use, and its cz loci."""
    sent = C.sent(spec.hardware)
    qubits = sorted({q for i in sent for q in i.locus}, key=dev.qubit_sort_key)
    cz = sorted({dev.pair_key(*i.locus) for i in sent if i.name == "cz"},
                key=lambda p: (dev.qubit_sort_key(p[0]), dev.qubit_sort_key(p[1])))
    return qubits, cz


def live_checks(fz: Frozen, jobs: Sequence[PlannedJob], snapshot: Path) -> dict[str, Any]:
    """The frozen circuits against a live snapshot's calibration: nothing rebuilt. Raises ``AssertionError`` on a lost gate."""
    snap = load_snapshot(fz.device, snapshot)
    arch = snap["architecture"]
    live_cal = live_calibration(fz.device, snapshot)
    live = {frozenset(k) for k in live_cal.couplers}
    lost = {}
    for job in jobs:
        for name in job.names:
            spec = fz.suite[name]
            qubits, cz = circuit_gates(spec)
            missing = {
                "prx": [q for q in qubits if not _served(arch, "prx", (q,))],
                "measure": [q for q in qubits if not _served(arch, "measure", (q,))],
                "cz": [f"{a}-{b}" for a, b in cz if not _served(arch, "cz", (a, b)) or frozenset((a, b)) not in live],
            }
            if fz.device == "emerald":
                try:
                    EM.check_live_couplers(spec, live_cal)
                except AssertionError as e:
                    missing["check_live_couplers"] = [str(e)]
            if any(missing.values()):
                lost[name] = missing
    assert not lost, f"STOP: the live calibration lost a gate a frozen circuit needs: {lost}"
    return {"snapshot": _rel(snapshot, REPO), "calibration_set_id": snap["calibration_set_id"],
            "same_as_design": snap["calibration_set_id"] == fz.conf.design_calset,
            "circuits": sum(len(j.names) for j in jobs), "lost": {}}


def rerank_info(fz: Frozen, snapshot: Path, shots: int = 50_000) -> dict[str, Any]:
    """For information only: the E1 placements re-ranked under a live calibration (the circuits stay frozen)."""
    from dqiqm.experiments import memory

    live_cal = live_calibration(fz.device, snapshot)
    arch = EM.architecture(live_cal) if fz.device == "emerald" else fz.suite.arch
    ranked = memory.rank(arch, live_cal, shots=shots)
    patches = {s.params["patch"]: s for s in fz.suite.of("E1")}
    frozen = {}
    for patch, s in sorted(patches.items()):
        entry = fz.placement[s.name]
        frozen[patch] = next((i for i, r in enumerate(ranked) if r["placement"].orientation == entry["orientation"]
                              and list(r["placement"].origin) == list(entry["origin"])
                              and frozenset(r["placement"].physical.values()) == frozenset(entry["placement"].values())),
                             None)
    return {"calibration_set_id": live_cal.calibration_set_id, "rank_shots": shots, "placements": len(ranked),
            "frozen_patch_rank": frozen,
            "top": [{"rank": i, "orientation": r["placement"].orientation, "origin": list(r["placement"].origin),
                     "mean_p_l_r3": r["mean_p_l"]} for i, r in enumerate(ranked[:6])],
            "note": "information only: the circuits stay frozen (emerald-design review, MINOR 1)"}


def run_request_checks(rr: Any, fz: Frozen, job: PlannedJob, calset: str) -> dict[str, Any]:
    """The run request ``hw.submit`` would build for a job, circuit by circuit, against the plan. Raises ``AssertionError``."""
    from iqm.iqm_client.models import DDMode, HeraldingMode

    assert len(rr.circuits) == len(job.names), (len(rr.circuits), len(job.names))
    sent = hw._sent_qubits(rr)  # noqa: SLF001 -- what the ledger row will record
    for i, name in enumerate(job.names):
        sha = instructions_sha256(rr.circuits[i].instructions)
        assert sha == fz.placement[name]["native_sha256"], f"position {i}, {name}: the run request's instructions hash to {sha}"
        assert sent[i] == fz.planned_qubits(name), f"position {i}, {name}: sent to {sent[i]}"
    assert rr.qubit_mapping is None, rr.qubit_mapping
    assert str(rr.calibration_set_id) == calset, rr.calibration_set_id
    assert rr.shots == job.shots, rr.shots
    assert rr.heralding_mode == HeraldingMode.NONE and rr.dd_mode == DDMode.DISABLED
    assert rr.active_reset_cycles is None
    return {"label": job.label, "circuits": len(rr.circuits), "shots": rr.shots, "calibration_set_id": str(rr.calibration_set_id),
            "qubits": sorted({q for s in sent for q in s}, key=dev.qubit_sort_key),
            "instructions_sha256_equal": True, "qubit_mapping": None,
            "options": {k: str(getattr(rr, k)) for k in ("heralding_mode", "dd_mode", "dd_strategy", "active_reset_cycles",
                                                          "max_circuit_duration_over_t2", "move_gate_validation",
                                                          "move_gate_frame_tracking")}}


def live_run_request(fz: Frozen, job: PlannedJob, calset: str) -> Any:
    """``create_run_request`` exactly as ``hw.submit`` calls it, on the read-only backend (it cannot submit)."""
    b = hw.backend(fz.device, calset)
    assert str(b.architecture.calibration_set_id) == calset, b.architecture.calibration_set_id
    return b.create_run_request([fz.suite[n].hardware.circuit for n in job.names], shots=job.shots,
                                qubit_index_to_name=fz.names_map(job))


# ---------------------------------------------------------------------------
# The decoder priors (the pilots'), and the sanity references.
# ---------------------------------------------------------------------------


PRIORS = "hardware_noise (79092cf)"


def decoder_priors(cal: dev.Calibration, spec: Any) -> tuple[Any, dict[str, Any], str] | None:
    """The decoder priors for a circuit: (noisy copy, ``hardware_noise``'s info, note), or None without an observable.

    ``device.hardware_noise(cal, spec)`` with its defaults, on either device (the orchestrator's
    instruction of 2026-10-25, replacing the brief's pilot priors).
    """
    if spec.noisy is None or not spec.stim.num_observables:
        return None
    nm, info = dev.hardware_noise(cal, spec)
    note = (f"priors: {PRIORS}. dqiqm.experiments.device.hardware_noise(calibration set {cal.calibration_set_id}, spec): "
            f"calibrated_noise with the {info['idle_channel']} idle over {info['idle_ns']:g} ns per mid-circuit readout "
            f"layer (readout window {info['readout_window_ns']:g} ns from {info['readout_window_source']}, plus a "
            f"{info['delay_ns']:g} ns delay), T2 {info['t2']}, final readout {info['final_readout']}")
    return spec.noisy(nm), info, note


def first_readouts(spec: Any) -> list[int]:
    """The column of each non-data qubit's first measurement (round 1's ancilla readout; E3's first layer)."""
    meas_q = C.measured_qubits(spec.stim)
    cols = {}
    for j, q in enumerate(meas_q):
        if spec.roles.get(q, "data") != "data":
            cols.setdefault(q, j)
    return sorted(cols.values())


def reference(device: str, spec: Any, cal: dev.Calibration | None, *, model: bool = True) -> dict[str, Any]:
    """What a sane run of ``spec`` looks like: the first readouts' pattern from a noiseless sample, and the model.

    A first readout is "random" if the noiseless circuit gives both values, else "deterministic" with the
    value of Stim's reference sample (the one detection events are taken against).
    """
    noiseless = spec.stim.compile_sampler(seed=SEED).sample(REF_SHOTS).astype(np.uint8)
    ref = np.array(spec.stim.reference_sample(), dtype=np.uint8)
    cols = first_readouts(spec)
    random_cols = [j for j in cols if 0 < noiseless[:, j].mean() < 1]
    det_cols = [j for j in cols if j not in random_cols]
    assert all((noiseless[:, j] == ref[j]).all() for j in det_cols), f"{spec.name}: a deterministic readout off the reference"
    every_deterministic = all(0 in (noiseless[:, j].mean(), 1 - noiseless[:, j].mean()) for j in range(noiseless.shape[1]))
    out: dict[str, Any] = {
        "circuit": spec.name, "measurements": spec.stim.num_measurements, "detectors": spec.stim.num_detectors,
        "first_cols": cols, "random_cols": random_cols, "deterministic_cols": det_cols,
        "deterministic_values": [int(ref[j]) for j in det_cols],
        "noiseless_all_zero": float((~noiseless.any(axis=1)).mean()),
        "noiseless_always_all_zero": bool(every_deterministic and not ref.any()),
        "x_type_first_cols": [], "z_type_first_cols": [],
    }
    if spec.stabilizers:
        meas_q = C.measured_qubits(spec.stim)
        for j in cols:
            st = spec.stabilizers.get(meas_q[j])
            if st is not None:
                out["x_type_first_cols" if st.pauli == "X" else "z_type_first_cols"].append(j)
    out["model"] = None
    if model and cal is not None:
        pri = decoder_priors(cal, spec)
        if pri is not None:
            noisy, info, _ = pri
            idle = info["idle_ns"]
            raw = noisy.compile_sampler(seed=SEED).sample(MODEL_SHOTS).astype(np.uint8)
            dets, _ = D.detection_events(spec.stim, raw)
            out["model"] = {"shots": MODEL_SHOTS, "seed": SEED, "idle_ns": idle, "detection_fraction": float(dets.mean()),
                            "all_zero": float((~raw.any(axis=1)).mean()),
                            "random_first_mean": float(raw[:, random_cols].mean()) if random_cols else None,
                            "deterministic_first_misread": float((raw[:, det_cols] != ref[det_cols]).mean()) if det_cols else None}
    return out


def sanity(raw: np.ndarray, dets: np.ndarray, ref: Mapping[str, Any], *, shots: int,
           physical: Any = None, planned: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The brief's sanity checks on one circuit's record: each with its value, band and pass."""
    raw = np.asarray(raw)
    checks: list[dict[str, Any]] = []

    def add(name: str, value: Any, band: Any, ok: bool, info: Any = None) -> None:
        checks.append({"check": name, "value": value, "band": band, "pass": bool(ok), **({"info": info} if info else {})})

    shaped = raw.ndim == 2 and raw.shape == (shots, ref["measurements"])
    add("measurement shape", list(raw.shape), [shots, ref["measurements"]], shaped)
    if not shaped:
        for name in ("all-zero rows", "first readouts, random", "first readouts, deterministic", "detection fraction"):
            add(name, None, "not computed: wrong shape", False)
    else:
        zeros = int((~raw.any(axis=1)).sum())
        if ref["noiseless_always_all_zero"]:
            add("all-zero rows", zeros, "any: the noiseless record is all zero", True)
        else:
            m = ref.get("model") or {}
            expect = max(ref["noiseless_all_zero"], m.get("all_zero") or 0.0)
            allowed = max(ALL_ZERO_PER_1000 * shots / 1000, 2 * expect * shots + 5)
            add("all-zero rows", zeros, f"<= {allowed:g}", zeros <= allowed)
        rc, dc = ref["random_cols"], ref["deterministic_cols"]
        if rc:
            x = float(raw[:, rc].mean())
            add("first readouts, random", x, list(RANDOM_BAND), RANDOM_BAND[0] <= x <= RANDOM_BAND[1],
                {"columns": rc, "x_type": [c for c in rc if c in ref["x_type_first_cols"]],
                 "z_type": [c for c in rc if c in ref["z_type_first_cols"]]})
        else:
            add("first readouts, random", None, "none in this circuit", True)
        if dc:
            want = np.asarray(ref["deterministic_values"], dtype=np.uint8)
            x = float((raw[:, dc] != want).mean())
            add("first readouts, deterministic (misread)", x, f"< {DETERMINISTIC_MAX}", x < DETERMINISTIC_MAX,
                {"columns": dc, "x_type": [c for c in dc if c in ref["x_type_first_cols"]],
                 "z_type": [c for c in dc if c in ref["z_type_first_cols"]]})
        else:
            add("first readouts, deterministic (misread)", None, "none in this circuit", True)
        f = float(np.asarray(dets).mean()) if np.asarray(dets).size else 0.0
        model = (ref.get("model") or {}).get("detection_fraction")
        limit = DETECTION_MAX_HIGH if model is not None and model >= DETECTION_HIGH_MODEL else DETECTION_MAX
        add("detection fraction", f, f"< {limit}", f < limit,
            {"model_with_idle": model, "ratio_to_model": f / model if model else None})
    if planned is not None:
        add("physical qubits (meta.json)", physical, list(planned), physical == list(planned))
    return checks


# ---------------------------------------------------------------------------
# The ledger, the manifests, the guards.
# ---------------------------------------------------------------------------


def purpose(seg: Segment, job: PlannedJob) -> str:
    return f"{seg.task} {job.label} {job.experiment} ({len(job.names)} circuits x {job.shots} shots)"


def ledger_rows(ledger: Path, task: str | None = None) -> list[hw.LedgerRow]:
    if not Path(ledger).is_file():
        return []
    return [r for r in hw.read_ledger(ledger) if task is None or r.task == task]


def label_of(row: hw.LedgerRow) -> str | None:
    words = row.purpose.split()
    return words[1] if len(words) > 1 and words[0] == row.task else None


def job_rows(ledger: Path, seg: Segment) -> dict[str, list[hw.LedgerRow]]:
    """The segment's ledger rows by job label (from their purpose)."""
    out: dict[str, list[hw.LedgerRow]] = defaultdict(list)
    for r in ledger_rows(ledger, seg.task):
        out[label_of(r) or "?"].append(r)
    return dict(out)


def check_arming(seg: Segment, env: Mapping[str, str]) -> None:
    """Refuse unless armed exactly as the segment's row of the brief, against the default ledger."""
    for k, v in seg.arming().items():
        if env.get(k) != v:
            raise SystemExit(f"refusing: {k}={env.get(k)!r}, the brief arms {seg.task} with {k}={v}")
    if hw.ledger_path(env) != Path(hw.DEFAULT_LEDGER):
        raise SystemExit(f"refusing: ledger {hw.ledger_path(env)}, not {hw.DEFAULT_LEDGER}")


def check_order(seg: Segment, ledger: Path) -> None:
    """Refuse unless every earlier segment has all its jobs in the ledger (the brief's exact order)."""
    for task in ORDER[: ORDER.index(seg.task)]:
        n = len(ledger_rows(ledger, task))
        if n != SEGMENTS[task].jobs:
            raise SystemExit(f"refusing: {task} comes before {seg.task} and has {n} of {SEGMENTS[task].jobs} jobs in the ledger")
    for task in ORDER[ORDER.index(seg.task) + 1:]:
        if ledger_rows(ledger, task):
            raise SystemExit(f"refusing: {task} comes after {seg.task} and already has ledger rows")


def find_job_dir(raw_root: Path, task: str, job_id: str) -> Path | None:
    """The job's raw directory, whatever day it landed on (``raw/<YYYYMMDD>-<task>/<job id>/``)."""
    dirs = [p.parent for p in Path(raw_root).glob(f"*-{task}/{job_id}/meta.json")]
    if len(dirs) > 1:
        raise SystemExit(f"job {job_id} is saved twice: {dirs}")
    return dirs[0] if dirs else None


def manifest_path(raw_root: Path, seg: Segment, job: PlannedJob, day: str) -> Path:
    return Path(raw_root) / f"{day}-{seg.task}" / f"{job.label}.json"


def job_manifest(fz: Frozen, seg: Segment, job: PlannedJob, calset: str) -> dict[str, Any]:
    """Position -> circuit name, ``native_sha256`` and qubits: what ``hw.submit`` is about to send."""
    return {
        "format": "dqiqm.batches/1", "task": seg.task, "device": seg.device, "label": job.label, "batch": seg.batch,
        "split": seg.split, "experiment": job.experiment, "segment_position": job.position,
        "shots": job.shots, "circuits": len(job.names), "calibration_set_id": calset, "purpose": purpose(seg, job),
        "order": [{"position": i, "circuit": n, "native_sha256": fz.placement[n]["native_sha256"],
                   "physical_qubits": fz.planned_qubits(n)} for i, n in enumerate(job.names)],
        "qubit_index_to_name": {str(k): v for k, v in fz.names_map(job).items()},
        "note": "IQM returns circuit_names [null]: this file is the record of which circuit is c<i> in memory.npz",
        "job_id": None, "raw": None, "written_utc": utc_now(),
    }


class LedgerWatch:
    """Print a new job id of ``task`` as soon as ``hw.submit`` writes its ledger row: ``garnet_pilot.LedgerWatch``, per task."""

    def __init__(self, ledger: Path, task: str, interval: float = 0.5) -> None:
        self.ledger, self.task, self.interval = Path(ledger), task, interval
        self.seen = {r.job_id for r in ledger_rows(self.ledger, task)}
        self.job_id: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _poll(self) -> None:
        try:
            rows = ledger_rows(self.ledger, self.task)
        except Exception:  # noqa: BLE001 -- a half-written line: read again next time
            return
        for r in rows:
            if r.job_id not in self.seen and self.job_id is None:
                self.job_id = r.job_id
                log(f"JOB ID {r.job_id}  (ledger row written {r.time}; waiting for the result)")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._poll()

    def __enter__(self) -> LedgerWatch:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()
        self._poll()


def check_ledger_row(row: hw.LedgerRow, *, seg: Segment, job: PlannedJob, job_id: str, calset: str,
                     qubits: Sequence[str]) -> dict[str, Any]:
    want = {"task": seg.task, "device": seg.device, "calibration_set_id": calset, "qubits": ",".join(qubits),
            "circuits x shots": f"{len(job.names)} x {job.shots}", "job id": job_id, "purpose": purpose(seg, job)}
    got = {"task": row.task, "device": row.device, "calibration_set_id": row.calibration_set_id, "qubits": row.qubits,
           "circuits x shots": f"{row.circuits} x {row.shots}", "job id": row.job_id, "purpose": row.purpose}
    return {k: {"want": want[k], "got": got[k], "ok": want[k] == got[k]} for k in want}


def meta_checks(meta: Mapping[str, Any], fz: Frozen, job: PlannedJob, calset: str) -> dict[str, bool]:
    return {
        "qubit_mapping": meta["qubit_mapping"] == [{str(k): v for k, v in sorted(m.items())} for m in fz.mappings(job)],
        "physical_qubits": meta["physical_qubits"] == [fz.planned_qubits(n) for n in job.names],
        "calibration_set_id": meta["calibration_set_id"] == calset,
        "shots": meta["shots"] == job.shots,
        "circuits": meta["circuits"] == len(job.names),
        "success": bool(meta["success"]),
        "timeline": bool(meta.get("timeline")),
    }


def job_sanity(job_dir: Path, fz: Frozen, job: PlannedJob, refs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every circuit of a saved job through :func:`sanity`."""
    meta = json.loads((Path(job_dir) / "meta.json").read_text())
    out = []
    for i, name in enumerate(job.names):
        spec = fz.suite[name]
        try:
            raw, _ = D.load_memory(job_dir, i, circuit=spec.stim)
        except (ValueError, KeyError, IndexError) as e:
            out.append({"position": i, "circuit": name, "pass": False, "error": f"{type(e).__name__}: {e}", "checks": []})
            continue
        dets, _ = D.detection_events(spec.stim, raw)
        phys = (meta.get("physical_qubits") or [None] * len(job.names))[i]
        checks = sanity(raw, dets, refs[name], shots=job.shots, physical=phys, planned=fz.planned_qubits(name))
        out.append({"position": i, "circuit": name, "pass": all(c["pass"] for c in checks), "checks": checks})
    return out


def sanity_line(results: Sequence[Mapping[str, Any]]) -> str:
    """One line: circuits passed, and the ranges of the detection fraction and first-readout figures."""
    def vals(name: str) -> list[float]:
        return [c["value"] for r in results for c in r["checks"] if c["check"].startswith(name) and c["value"] is not None]

    det, rnd, dtm = vals("detection fraction"), vals("first readouts, random"), vals("first readouts, deterministic")
    span = lambda v: f"{min(v):.3f}-{max(v):.3f}" if v else "n/a"  # noqa: E731
    return (f"{sum(r['pass'] for r in results)}/{len(results)} circuits pass; detection fraction {span(det)}; "
            f"random first readouts {span(rnd)}; deterministic misread {span(dtm)}")


# ---------------------------------------------------------------------------
# snapshot, verify
# ---------------------------------------------------------------------------


def _snapshot_files() -> set[Path]:
    return {p for d in ("garnet", "emerald") for p in (CALIBRATION / d).glob("*.json")}


def take_snapshot(device: str) -> Path:
    """Run ``scripts/resonance_check.py`` live (read-only) and return the device's new snapshot."""
    before = _snapshot_files()
    log(f"{utc_now()} running scripts/resonance_check.py live (read-only)")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "resonance_check.py")], cwd=REPO,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    new = sorted(_snapshot_files() - before)
    for p in new:
        log(f"new snapshot: {p.relative_to(REPO)}")
    mine = [p for p in new if p.parent.name == device]
    if proc.returncode != 0 or len(mine) != 1:
        log(proc.stdout[-4000:])
        raise SystemExit(f"STOP: resonance_check.py exited {proc.returncode}, {len(mine)} new {device} snapshot(s)")
    return mine[0]


def cmd_snapshot(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    path = take_snapshot(seg.device)
    snap = load_snapshot(seg.device, path)
    log(f"SNAPSHOT {path.relative_to(REPO)}")
    log(f"  calibration_set_id {snap['calibration_set_id']}; server default at fetch {snap.get('default_calibration_set_id_at_fetch')}")
    log(f"  fetched {snap['fetched_utc_start']} .. {snap['fetched_utc_end']}")
    log(f"  same as the design's {DEVICES[seg.device].design_calset}: {snap['calibration_set_id'] == DEVICES[seg.device].design_calset}")
    return 0


def references(fz: Frozen, jobs: Sequence[PlannedJob], cal: dev.Calibration) -> dict[str, dict[str, Any]]:
    return {n: reference(fz.device, fz.suite[n], cal) for job in jobs for n in job.names}


def preflight(fz: Frozen, seg: Segment, snapshot: Path | None, *, live: bool, rerank: bool = False) -> dict[str, Any]:
    """Everything before the first job: the plan, the freeze, the live checks, the pinned requests, the references."""
    jobs = fz.jobs(seg)
    out: dict[str, Any] = {"task": seg.task, "device": seg.device, "batch": seg.batch, "split": seg.split,
                           "jobs": [{"label": j.label, "experiment": j.experiment, "circuits": len(j.names), "shots": j.shots,
                                     "total_shots": j.total_shots, "order": list(j.names)} for j in jobs],
                           "allowance": {"jobs": seg.jobs, "shots": seg.shots},
                           "frozen_from": _rel(fz.conf.committed_snapshot, REPO), "frozen_calset": fz.cal.calibration_set_id}
    out["freeze"] = {j.label: freeze_checks(fz, j) for j in jobs}
    log(f"FREEZE OK: {sum(len(j.names) for j in jobs)} circuits rebuilt from {out['frozen_from']} equal placement.json "
        "(native, stim and qasm hashes), native ops only, cz on the device's couplers, clbit j = measurement j, 0 SWAPs")
    for j in jobs:
        m = fz.names_map(j)
        log(f"  {j.label}: {len(j.names)} x {j.shots}; qubit_index_to_name over {len(m)} qubits "
            f"({','.join(sorted(m.values(), key=dev.qubit_sort_key))})")
    cal = fz.cal
    if snapshot is not None:
        out["live"] = live_checks(fz, jobs, snapshot)
        log(f"LIVE CALIBRATION OK in {snapshot}: {json.dumps(out['live'])}")
        cal = live_calibration(seg.device, snapshot)
        if rerank and not out["live"]["same_as_design"]:
            out["rerank"] = rerank_info(fz, snapshot)
            log("RE-RANK (information only, nothing rebuilt): " + json.dumps(out["rerank"]))
        if live:
            calset = out["live"]["calibration_set_id"]
            out["run_requests"] = {j.label: run_request_checks(live_run_request(fz, j, calset), fz, j, calset) for j in jobs}
            for label, rr in out["run_requests"].items():
                log(f"RUN REQUEST OK (read-only backend, not submitted) {label}: {rr['circuits']} circuits x {rr['shots']}, "
                    f"calset {rr['calibration_set_id']}, every circuit's instructions = its native_sha256, qubits as placed")
    out["references"] = references(fz, jobs, cal)
    out["references_calibration_set_id"] = cal.calibration_set_id
    return out


def cmd_plan(args: argparse.Namespace) -> int:
    total_jobs = total_shots = 0
    for seg in SEGMENTS.values():
        jobs = check_plan(seg, plan_json(seg.device))
        total_jobs += len(jobs)
        total_shots += sum(j.total_shots for j in jobs)
        log(f"{seg.task}: {seg.device} batch {seg.batch} ({seg.split}): "
            + ", ".join(f"{j.label} {len(j.names)}x{j.shots}" for j in jobs) + f" = {len(jobs)} jobs, {sum(j.total_shots for j in jobs):,} shots")
    log(f"total {total_jobs} jobs, {total_shots:,} shots (the brief: {TOTAL_JOBS}, {TOTAL_SHOTS:,})")
    return 0 if (total_jobs, total_shots) == (TOTAL_JOBS, TOTAL_SHOTS) else 1


def cmd_verify(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    if args.live and not args.snapshot:
        raise SystemExit("--live needs --snapshot (the calibration set to build the requests for)")
    fz = Frozen(seg.device)
    out = preflight(fz, seg, args.snapshot, live=args.live, rerank=args.rerank)
    for name, ref in out["references"].items():
        m = ref["model"] or {}
        log(f"  ref {name}: random first {ref['random_cols']} deterministic first {ref['deterministic_cols']} "
            f"noiseless all-zero {ref['noiseless_all_zero']:.4f}; model detection {m.get('detection_fraction')} "
            f"all-zero {m.get('all_zero')}")
    if args.out:
        write_json(Path(args.out), out)
        log(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# submit, recover
# ---------------------------------------------------------------------------


def submit_job(fz: Frozen, seg: Segment, job: PlannedJob, *, calset: str, ledger: Path, raw_root: Path,
               refs: Mapping[str, Mapping[str, Any]], timeout: float | None) -> dict[str, Any]:
    """One ``hw.submit`` of one job. Returns what happened; a failure after the job id says how to recover."""
    rows = job_rows(ledger, seg).get(job.label, [])
    if rows:
        raise SystemExit(f"refusing: {job.label} already has ledger row(s) {[r.job_id for r in rows]}; there is no resubmission")
    man = job_manifest(fz, seg, job, calset)
    mpath = manifest_path(raw_root, seg, job, dt.datetime.now(dt.UTC).strftime("%Y%m%d"))
    if mpath.exists() and json.loads(mpath.read_text()).get("job_id"):
        raise SystemExit(f"refusing: {mpath} already names a job")
    write_json(mpath, man)
    log(f"{utc_now()} manifest {_rel(mpath, REPO)} written; submitting {job.label}: {len(job.names)} circuits x {job.shots} "
        f"shots on {seg.device}, calibration set {calset}")
    error, sub = None, None
    with LedgerWatch(ledger, seg.task):
        try:
            sub = hw.submit([fz.suite[n].hardware.circuit for n in job.names], device=seg.device, shots=job.shots,
                            purpose=purpose(seg, job), calibration_set_id=calset, qubit_index_to_name=fz.names_map(job),
                            data_root=raw_root, timeout=timeout)
        except BaseException as e:  # noqa: BLE001 -- reported below, whatever it is
            error = e
    rows = job_rows(ledger, seg).get(job.label, [])
    job_id = rows[0].job_id if len(rows) == 1 else None
    man.update(job_id=job_id, submitted_utc=rows[0].time if rows else None)
    write_json(mpath, man)
    log(f"JOB ID {job_id}" if job_id else f"NO JOB ID: {len(rows)} ledger rows for {job.label}")
    if error is not None:
        log(f"hw.submit raised {type(error).__name__}: {error}")
        if job_id:
            log(f"RECOVER (read-only, never submit again): uv run python scripts/batch_run.py recover --segment {seg.task} "
                f"--job-id {job_id}")
        return {"label": job.label, "job_id": job_id, "error": f"{type(error).__name__}: {error}", "ok": False,
                "manifest": str(mpath)}
    return finish_job(fz, seg, job, Path(sub.path), calset=calset, ledger=ledger, refs=refs, manifest=mpath)


def finish_job(fz: Frozen, seg: Segment, job: PlannedJob, job_dir: Path, *, calset: str, ledger: Path,
               refs: Mapping[str, Mapping[str, Any]], manifest: Path) -> dict[str, Any]:
    """After the raw data is saved: the ledger row, ``meta.json`` and every circuit's sanity checks."""
    meta = json.loads((job_dir / "meta.json").read_text())
    rows = [r for r in ledger_rows(ledger, seg.task) if r.job_id == meta["job_id"]]
    qubits = sorted({q for n in job.names for q in fz.planned_qubits(n)}, key=dev.qubit_sort_key)
    row_check = check_ledger_row(rows[0], seg=seg, job=job, job_id=meta["job_id"], calset=calset, qubits=qubits) if rows else {}
    mcheck = meta_checks(meta, fz, job, calset)
    results = job_sanity(job_dir, fz, job, refs)
    ok_row = len(rows) == 1 and all(v["ok"] for v in row_check.values())
    ok = ok_row and all(mcheck.values()) and all(r["pass"] for r in results)
    man = json.loads(manifest.read_text()) if manifest.exists() else job_manifest(fz, seg, job, calset)
    man.update(job_id=meta["job_id"], raw=_rel(job_dir, fz.conf.record), ledger_rows=len(rows), ledger_check=row_check,
               meta_check=mcheck, sanity=results, ok=ok, checked_utc=utc_now())
    write_json(manifest, man)
    log(f"raw: {_rel(job_dir, REPO)}")
    log(f"ledger row ({len(rows)}): " + ("OK" if ok_row else json.dumps(row_check)))
    log("meta.json: " + ("OK" if all(mcheck.values()) else json.dumps(mcheck)))
    for r in results:
        if not r["pass"]:
            log(f"  SANITY FAIL position {r['position']} {r['circuit']}: "
                + json.dumps([c for c in r["checks"] if not c["pass"]] or r.get("error")))
    log(f"sanity: {sanity_line(results)}")
    return {"label": job.label, "job_id": meta["job_id"], "path": str(job_dir), "ledger_rows": len(rows), "ok": ok,
            "manifest": str(manifest), "sanity": sanity_line(results)}


def resume_state(fz: Frozen, seg: Segment, job: PlannedJob, *, ledger: Path, raw_root: Path,
                 refs: Mapping[str, Mapping[str, Any]]) -> str:
    """A job's state before ``submit`` reaches it: "new", "done" (in the ledger, saved, checked OK), "failed" or
    "raw missing". A job saved but never checked (the process died in between) is checked now."""
    done = job_rows(ledger, seg).get(job.label, [])
    if not done:
        return "new"
    jd = find_job_dir(raw_root, seg.task, done[0].job_id)
    log(f"{job.label}: already submitted as {done[0].job_id}" + ("" if jd else " -- RAW MISSING: run recover"))
    if jd is None:
        return "raw missing"
    mpath = manifest_path(raw_root, seg, job, done[0].time[:10].replace("-", ""))
    checked = json.loads(mpath.read_text()).get("ok") if mpath.exists() else None
    if checked is None:
        checked = finish_job(fz, seg, job, jd, calset=done[0].calibration_set_id, ledger=ledger, refs=refs, manifest=mpath)["ok"]
    return "done" if checked else "failed"


def cmd_submit(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    check_arming(seg, os.environ)
    ledger = hw.ledger_path()
    check_order(seg, ledger)
    snapshot = Path(args.snapshot)
    snap = load_snapshot(seg.device, snapshot)
    calset = snap["calibration_set_id"]
    fz = Frozen(seg.device)
    raw_root = fz.conf.raw_root
    pre = preflight(fz, seg, snapshot, live=False)
    refs = pre["references"]
    jobs = fz.jobs(seg)
    day = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    pre_path = raw_root / f"{day}-{seg.task}" / "preflight.json"
    if not pre_path.exists():
        write_json(pre_path, pre)
    default_now = hw.calibration_set_id(seg.device)
    if default_now != calset:
        raise SystemExit(f"refusing: the server's default calibration set is now {default_now}, the snapshot's {calset}; "
                         "take a new snapshot")
    results, calsets = [], [{"calibration_set_id": calset, "snapshot": _rel(snapshot, REPO), "from_job": jobs[0].label}]
    for job in jobs:
        state = resume_state(fz, seg, job, ledger=ledger, raw_root=raw_root, refs=refs)
        if state == "raw missing":
            return 5
        if state == "failed":
            log(f"STOP: {job.label} failed its checks; the rest of {seg.task} is not submitted. Report it.")
            return 4
        if state == "done":
            continue
        now = hw.calibration_set_id(seg.device)
        if now != calset:
            log(f"{utc_now()} the server's default calibration set moved: {calset} -> {now}; a new snapshot and the live checks")
            new = take_snapshot(seg.device)
            try:
                live = live_checks(fz, jobs, new)
            except AssertionError as e:
                log(str(e))
                return 6
            if live["calibration_set_id"] != now:
                raise SystemExit(f"STOP: the new snapshot is {live['calibration_set_id']}, the server default {now}")
            calset = now
            refs = references(fz, jobs, live_calibration(seg.device, new))
            calsets.append({"calibration_set_id": calset, "snapshot": _rel(new, REPO), "from_job": job.label})
        try:
            rr = run_request_checks(live_run_request(fz, job, calset), fz, job, calset)
        except AssertionError as e:
            log(f"STOP: the run request for {job.label} failed its check: {e}")
            return 7
        log(f"RUN REQUEST OK (read-only backend, not submitted) {job.label}: {json.dumps(rr)}")
        res = submit_job(fz, seg, job, calset=calset, ledger=ledger, raw_root=raw_root, refs=refs, timeout=args.timeout)
        results.append(res)
        log(f"{job.label} " + ("OK" if res["ok"] else "NOT OK") + ": " + json.dumps(res))
        if not res["ok"]:
            log(f"STOP: {job.label} failed a check; the rest of {seg.task} is not submitted. Report it.")
            return 4
    rows = job_rows(ledger, seg)
    seg_file = raw_root / f"{day}-{seg.task}" / "segment.json"
    write_json(seg_file, {
        "format": "dqiqm.batches/1", "task": seg.task, "device": seg.device, "batch": seg.batch, "split": seg.split,
        "allowance": {"jobs": seg.jobs, "shots": seg.shots}, "calibration_sets": calsets, "before": _rel(snapshot, REPO),
        "jobs": [{"label": j.label, "position": j.position, "job_id": rows[j.label][0].job_id, "ledger_time": rows[j.label][0].time,
                  "calibration_set_id": rows[j.label][0].calibration_set_id,
                  "raw": _rel(find_job_dir(raw_root, seg.task, rows[j.label][0].job_id), fz.conf.record),
                  "order": [{"position": i, "circuit": n, "native_sha256": fz.placement[n]["native_sha256"]}
                            for i, n in enumerate(j.names)]} for j in jobs],
        "written_utc": utc_now()})
    n = len(ledger_rows(ledger, seg.task))
    shots = sum(r.total_shots for r in ledger_rows(ledger, seg.task))
    log(f"SEGMENT SUBMITTED {seg.task}: {n} jobs, {shots:,} shots in the ledger (allowance {seg.jobs}, {seg.shots:,}); "
        f"{_rel(seg_file, REPO)}")
    return 0 if (n, shots) == (seg.jobs, seg.shots) else 8


def recover_job(fz: Frozen, seg: Segment, job_id: str, *, ledger: Path, raw_root: Path, timeout: float | None) -> tuple[PlannedJob, Path]:
    """Re-pull a job read-only and save it in ``save_raw``'s layout, with the submission's qubits. Never submits."""
    rows = [r for r in ledger_rows(ledger, seg.task) if r.job_id == job_id]
    if len(rows) != 1:
        raise SystemExit(f"refusing: job {job_id} has {len(rows)} {seg.task} rows in {ledger}")
    label = label_of(rows[0])
    job = next((j for j in fz.jobs(seg) if j.label == label), None)
    if job is None:
        raise SystemExit(f"refusing: the ledger row's purpose names {label!r}, not a job of {seg.task}")
    day = rows[0].time[:10].replace("-", "")
    target = Path(raw_root) / f"{day}-{seg.task}" / job_id
    if target.exists():
        raise SystemExit(f"refusing: {target} exists ({sorted(x.name for x in target.iterdir())}); it is never overwritten")
    result = hw.retrieve(job_id, seg.device, timeout=timeout)
    path = hw.save_raw(result, task=seg.task, device=seg.device, qubit_mapping=fz.mappings(job), shots=job.shots,
                       physical_qubits=[fz.planned_qubits(n) for n in job.names], purpose=rows[0].purpose, day=day,
                       data_root=raw_root)
    return job, path


def cmd_recover(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    fz = Frozen(seg.device)
    ledger = hw.ledger_path()
    job, path = recover_job(fz, seg, args.job_id, ledger=ledger, raw_root=fz.conf.raw_root, timeout=args.timeout)
    log(f"RECOVERED (hw.retrieve + hw.save_raw, read-only): {path}")
    row = next(r for r in ledger_rows(ledger, seg.task) if r.job_id == args.job_id)
    cal = live_calibration(seg.device, Path(args.snapshot)) if args.snapshot else fz.cal
    refs = references(fz, [job], cal)
    day = row.time[:10].replace("-", "")
    res = finish_job(fz, seg, job, path, calset=row.calibration_set_id, ledger=ledger, refs=refs,
                     manifest=manifest_path(fz.conf.raw_root, seg, job, day))
    log(f"{job.label} " + ("OK" if res["ok"] else "NOT OK") + ": " + json.dumps(res))
    return 0 if res["ok"] else 4


# ---------------------------------------------------------------------------
# analyze: the records
# ---------------------------------------------------------------------------


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def timeline_numbers(timeline: Sequence[Mapping[str, Any]], *, shots: int) -> dict[str, Any]:
    """Spans of the server's status timeline (``garnet_pilot.timeline_numbers``, without the per-circuit plan)."""
    first: dict[str, dt.datetime] = {}
    for e in timeline:
        try:
            first.setdefault(str(e["status"]), _ts(e["timestamp"]))
        except (TypeError, KeyError, ValueError):
            continue
    out: dict[str, Any] = {"statuses": list(first)}
    if first:
        out["total_s"] = (max(first.values()) - min(first.values())).total_seconds()
        out["created"] = min(first.values()).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    for name, a, b in (("queue_s", "received", "execution_started"), ("compile_s", "compilation_started", "compilation_ended"),
                       ("wait_s", "compilation_ended", "execution_started"),
                       ("execution_s", "execution_started", "execution_ended"), ("post_s", "execution_ended", "completed")):
        if a in first and b in first:
            out[name] = (first[b] - first[a]).total_seconds()
    for k in ("execution_started", "execution_ended", "completed"):
        if k in first:
            out[k] = first[k].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if "execution_s" in out and shots:
        out["execution_per_shot_us"] = out["execution_s"] / shots * 1e6
    return out


def ler_group(spec: Any, batch: int) -> tuple[str, str, str] | None:
    """(experiment, LER run id, title) as the dry runs group them; None for E3."""
    p = spec.params
    if spec.experiment == "E1":
        return "E1", f"{p['basis']}_{p['patch']}_b{batch}", f"E1 d=3 {p['basis']} basis, patch {p['patch']}, batch {batch}"
    if spec.experiment == "E4":
        return ("E4", f"{p['basis']}_{p['patch']}_t{int(p['delay_ns'])}ns_b{batch}",
                f"E4 d=3 {p['basis']}, patch {p['patch']}, {p['delay_ns'] / 1e3:g} us idle between rounds, batch {batch}")
    if spec.experiment == "E2":
        return ("E2", f"{p['kind']}_{'p' if p['prep'] == '+' else p['prep']}_b{batch}",
                f"E2 repetition d=10, {p['kind']}-flip, |{p['prep']}>, batch {batch}")
    if spec.experiment == "E5":
        return "E5", f"{p['kind']}_b{batch}", f"E5 stability ({p['kind']} edge checks), batch {batch}"
    return None


def per_round(dets: np.ndarray, t: np.ndarray) -> dict[str, float]:
    return {str(k): float(dets[:, t == k].mean()) for k in sorted(set(t.tolist()))}


def analyze(seg: Segment, snapshots: Sequence[Path], *, root: Path, ledger: Path, fz: Frozen | None = None) -> dict[str, Any]:
    """Write the segment's records: every run's dataset, derived features and checks, then its LER records."""
    fz = fz or Frozen(seg.device)
    jobs = fz.jobs(seg)
    rows = job_rows(ledger, seg)
    cals = {}
    for p in snapshots:
        c = live_calibration(seg.device, Path(p))
        cals.setdefault(c.calibration_set_id, (Path(p), c))
    snap_rel = {"before": _rel(Path(snapshots[0]), root), "after": _rel(Path(snapshots[-1]), root)}
    label = "HARDWARE"
    runs, job_out = [], []
    for job in jobs:
        jr = rows.get(job.label, [])
        if len(jr) != 1:
            raise SystemExit(f"{job.label}: {len(jr)} ledger rows")
        job_dir = find_job_dir(fz.conf.raw_root, seg.task, jr[0].job_id)
        if job_dir is None:
            raise SystemExit(f"{job.label}: job {jr[0].job_id} has no raw data; recover it first")
        meta = json.loads((job_dir / "meta.json").read_text())
        if meta["calibration_set_id"] not in cals:
            raise SystemExit(f"{job.label} ran on {meta['calibration_set_id']}; pass its snapshot (have {sorted(cals)})")
        snap_path, cal = cals[meta["calibration_set_id"]]
        durations = dev.Durations.from_calibration(cal)
        tl = timeline_numbers(meta.get("timeline") or [], shots=job.total_shots)
        job_out.append({"label": job.label, "job_id": meta["job_id"], "calibration_set_id": meta["calibration_set_id"],
                        "circuits": len(job.names), "shots": job.shots, "total_shots": job.total_shots,
                        "ledger_time": jr[0].time, "timeline": tl, "raw": _rel(job_dir, root)})
        log(f"{job.label} {meta['job_id']}: {len(job.names)} circuits, timeline {json.dumps(tl)}")
        labels = SU.split(job.label, seg.batch, job.shots)
        job_checks = []
        for i, name in enumerate(job.names):
            spec = fz.suite[name]
            entry = fz.placement[name]
            raw, _ = D.load_memory(job_dir, i, circuit=spec.stim)
            run_id = f"b{seg.batch}_{name}"
            pri = decoder_priors(cal, spec)
            dec, prior_info, note = pri if pri is not None else (None, None, "")
            files = {"stim": f"circuits/{spec.experiment}/{name}.stim", "qasm": f"circuits/{spec.experiment}/{name}.qasm",
                     "qasm_sha256": entry["qasm_sha256"], "native_sha256": entry["native_sha256"]}
            ds, numbers = IR.write_dataset(
                root, spec, raw, run_id=run_id, batch=seg.batch, job_id=meta["job_id"], position_in_job=i, arch=fz.suite.arch,
                cal=cal, durations=durations, circuit_files=files,
                raw_ref={"path": _rel(job_dir / "memory.npz", root), "key": f"c{i}", "meta": _rel(job_dir / "meta.json", root)},
                calibration_paths={**snap_rel, "job": _rel(snap_path, root)}, decoder_circuit=dec, decoder_note=note,
                split_labels=labels, split_info=SU.split_rule(job.label, seg.batch, labels), simulated=False,
                extra={"device": seg.device, "job_timeline": meta.get("timeline")},
            )
            der, summary = IR.write_derived(root, spec, run_id, ds)
            tags = {"label": label, "device": seg.device, "batch": seg.batch, "split": seg.split}
            if dec is not None:
                tags["priors"] = PRIORS
            tag_json(der / "summary.json", **tags)
            problems = IR.check_run(ds, der)
            dets = IR.load_array(ds, "dets")
            with open(ds / "detectors.csv") as f:
                t = np.array([int(r["round"]) for r in csv.DictReader(f)])
            ref = reference(seg.device, spec, cal)
            checks = sanity(raw, dets, ref, shots=job.shots, physical=(meta.get("physical_qubits") or [None] * 99)[i],
                            planned=fz.planned_qubits(name))
            body = {"format": IR.FORMAT, "label": label, "task": seg.task, "device": seg.device, "batch": seg.batch,
                    "split": seg.split, "job": job.label, "job_id": meta["job_id"], "position_in_job": i, "circuit": name,
                    "calibration_set_id": meta["calibration_set_id"], "priors": PRIORS if dec is not None else None,
                    "prior": prior_info, "sanity": checks, "sanity_pass": all(c["pass"] for c in checks),
                    "per_round_detection_rate": per_round(dets, t) if dets.size else {},
                    "model_with_idle": ref["model"], "numbers": numbers, "record_problems": problems,
                    "created_utc": utc_now(), "commit": hw.git_head()}
            write_json(der / "batch_checks.json", body)
            job_checks.append({"pass": body["sanity_pass"], "checks": checks})
            runs.append({"job": job.label, "spec": spec, "run_id": run_id, "dataset": ds, "derived": der, "numbers": numbers,
                         "problems": problems, "sanity_pass": body["sanity_pass"], "note": note})
            if problems or not body["sanity_pass"]:
                log(f"  {run_id}: problems {problems}; sanity {body['sanity_pass']}")
        job_out[-1]["sanity"] = sanity_line(job_checks)
        job_out[-1]["detection_fraction"] = [c["value"] for jc in job_checks for c in jc["checks"]
                                             if c["check"] == "detection fraction"]
        job_out[-1]["detection_ratio_to_model"] = [c["info"]["ratio_to_model"] for jc in job_checks for c in jc["checks"]
                                                   if c["check"] == "detection fraction" and c.get("info", {}).get("ratio_to_model")]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        g = ler_group(r["spec"], seg.batch)
        if g is not None and r["numbers"]["pm_failures"] is not None:
            groups[g].append(r)
    ler = []
    for (exp, run_id, title), rs in sorted(groups.items()):
        points = [{"circuit": r["spec"].name, "dataset": _rel(r["dataset"], root), "rounds": r["spec"].rounds,
                   "shots": r["numbers"]["shots"], "failures": r["numbers"]["pm_failures"],
                   "raw_failures": r["numbers"]["obs_flips"]} for r in rs]
        path, res = IR.write_ler(root, exp, run_id, points, title=f"{label} ({seg.device}): {title}", label=run_id,
                                 fit=exp != "E5")
        tag_json(path / "ler_fits.json", label=label, device=seg.device, batch=seg.batch, split=seg.split, priors=PRIORS,
                 decoder="pymatching on the noisy copy of the circuit that ran", noise=rs[0]["note"],
                 zero_failure_points=[x["rounds"] for x in points if x["failures"] == 0])
        ler.append({"experiment": exp, "run_id": run_id, "points": len(points), "shots": sum(x["shots"] for x in points),
                    **({"eps_fixed_A": res["fixed_A"]["eps"], "eps_free_A": res["free_A"]["eps"],
                        "amplitude": res["free_A"]["amplitude"]} if "free_A" in res else {})})
        if "free_A" in res:
            log(f"LER {exp} {run_id}: eps(A=1) {res['fixed_A']['eps']:.4f}, eps(free A) {res['free_A']['eps']:.4f}, "
                f"A {res['free_A']['amplitude']:.3f}")
    problems = {r["run_id"]: r["problems"] for r in runs if r["problems"]}
    out = {"task": seg.task, "device": seg.device, "batch": seg.batch, "split": seg.split, "jobs": job_out,
           "runs": len(runs), "record_problems": problems, "sanity_failures": [r["run_id"] for r in runs if not r["sanity_pass"]],
           "ler": ler, "snapshots": [_rel(Path(p), REPO) for p in snapshots], "created_utc": utc_now()}
    return out


def cmd_analyze(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    root = Path(args.root) if args.root else DEVICES[seg.device].record
    res = analyze(seg, [Path(p) for p in args.snapshots], root=root, ledger=Path(args.ledger))
    out = Path(args.out) if args.out else root / "ler" / "summary" / "segments" / f"{seg.task}.json"
    write_json(out, res)
    log(f"ANALYZED {seg.task}: {res['runs']} runs; record problems {res['record_problems'] or 'none'}; "
        f"sanity failures {res['sanity_failures'] or 'none'}; {len(res['ler'])} LER records; wrote {out}")
    return 0 if not res["record_problems"] and not res["sanity_failures"] else 1


# ---------------------------------------------------------------------------
# e3: the readout features, pooled per qubit
# ---------------------------------------------------------------------------


def e3_features(root: Path, batch: int) -> dict[str, Any]:
    """Per qubit, from batch ``batch``'s E3 runs: the assignment matrix and the record- and state-flip rates."""
    der = root / "derived" / "E3"
    assign: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    flips: dict[tuple[str, int], dict[str, float]] = {}
    runs = sorted(p for p in der.glob(f"b{batch}_E3_*") if p.is_dir())
    for run in runs:
        with open(run / "assignment.csv") as f:
            for row in csv.DictReader(f):
                if int(row["layer"]) != 0:
                    continue
                acc = assign[row["qb"]][int(row["prep"])]
                n = int(row["shots"])
                acc[0] += float(row["p_wrong"]) * n
                acc[1] += n
        if (run / "readout_flips.csv").exists() and ("_repeat_all0_" in run.name or "_repeat_all1_" in run.name):
            with open(run / "readout_flips.csv") as f:
                for row in csv.DictReader(f):
                    flips[(row["qb"], int(row["prep"]))] = {k: float(row[k]) for k in ("p_det_bulk", "record_flip", "state_flip",
                                                                                        "p_wrong_first", "p_wrong_last")}
    qubits = sorted(assign, key=dev.qubit_sort_key)
    rows = []
    for q in qubits:
        a0, a1 = assign[q].get(0, [0.0, 0]), assign[q].get(1, [0.0, 0])
        row = {"qb": q, "p1_given_0": a0[0] / a0[1] if a0[1] else None, "p0_given_1": a1[0] / a1[1] if a1[1] else None,
               "shots_prep0": a0[1], "shots_prep1": a1[1]}
        for prep in (0, 1):
            fl = flips.get((q, prep), {})
            row[f"record_flip_prep{prep}"] = fl.get("record_flip")
            row[f"state_flip_prep{prep}"] = fl.get("state_flip")
        rows.append(row)
    out_dir = der / f"b{batch}_pooled"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "readout_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    def med(key: str) -> float | None:
        v = [r[key] for r in rows if r[key] is not None]
        return float(np.median(v)) if v else None

    summary = {"format": IR.FORMAT, "label": "HARDWARE", "device": "garnet", "batch": batch, "runs": [p.name for p in runs],
               "qubits": len(rows), "assignment": "layer-0 readouts of every E3 circuit, pooled per (qubit, prepared bit)",
               "flips": "derived/E3/<run>/readout_flips.csv of the repeat circuits (all0: prep 0, all1: prep 1; chkA/chkB "
                        "mix both and are not pooled here)",
               "median": {k: med(k) for k in ("p1_given_0", "p0_given_1", "record_flip_prep0", "record_flip_prep1",
                                              "state_flip_prep0", "state_flip_prep1")},
               "worst": {k: max(((r[k], r["qb"]) for r in rows if r[k] is not None), default=None)
                         for k in ("p1_given_0", "p0_given_1", "record_flip_prep1", "state_flip_prep1")},
               "rows": rows}
    write_json(out_dir / "summary.json", summary)
    return summary


def cmd_e3(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    if seg.device != "garnet":
        raise SystemExit("E3 runs on Garnet only")
    s = e3_features(DEVICES["garnet"].record, seg.batch)
    log(json.dumps({k: v for k, v in s.items() if k != "rows"}, default=str))
    for r in s["rows"]:
        log("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    return 0


# ---------------------------------------------------------------------------
# summary: pooled fits, X against Z, the doc's tables
# ---------------------------------------------------------------------------


def read_points(root: Path, experiment: str = "E1") -> list[dict[str, Any]]:
    """Every HARDWARE LER point of ``experiment`` under ``root/ler``, with its run's basis, patch and batch."""
    out = []
    for f in sorted((root / "ler" / experiment).glob("*_b[123]/ler_points.csv")):
        fits = json.loads((f.parent / "ler_fits.json").read_text())
        if fits.get("label") != "HARDWARE":
            continue
        with open(f) as fh:
            for row in csv.DictReader(fh):
                ds = json.loads((root / row["dataset"] / "manifest.json").read_text())
                out.append({"run_id": row["run_id"], "circuit": row["circuit"], "rounds": int(row["rounds"]),
                            "shots": int(row["shots"]), "failures": int(row["failures"]), "raw_failures": int(row["raw_failures"]),
                            "basis": ds["spec"]["basis"], "patch": ds["spec"]["patch"], "batch": int(ds["batch"]),
                            "delay_ns": ds["spec"].get("delay_ns") or 0.0})
    return out


def pool(points: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for p in points:
        by[p["rounds"]][0] += p["failures"]
        by[p["rounds"]][1] += p["shots"]
    r = np.array(sorted(by))
    return r, np.array([by[x][0] for x in r]), np.array([by[x][1] for x in r])


def fit_both(r: np.ndarray, k: np.ndarray, n: np.ndarray, bootstrap: int = 1000) -> dict[str, Any]:
    out: dict[str, Any] = {"rounds": r.tolist(), "failures": k.tolist(), "shots": n.tolist()}
    lo, hi = stats.wilson(k, n)
    out["p_l"] = (k / n).tolist()
    out["wilson95"] = [[float(a), float(b)] for a, b in zip(lo, hi)]
    if len(np.unique(r)) >= 2:
        fixed = stats.fit_eps(r, k, n, free_amplitude=False, bootstrap=bootstrap, seed=SEED)
        free = stats.fit_eps(r, k, n, free_amplitude=True, bootstrap=bootstrap, seed=SEED)
        out["fixed_A"] = fixed.to_dict() | {"chi2_per_dof": fixed.chi2_per_dof}
        out["free_A"] = free.to_dict() | {"chi2_per_dof": free.chi2_per_dof}
        out["_fits"] = (fixed, free)
    return out


def summary_fits(root: Path, device: str) -> dict[str, Any]:
    """E1 per (basis, patch): per batch, pooled finetune (B1+B2) and test (B3), both forms, bootstrap."""
    pts = read_points(root, "E1")
    out: dict[str, Any] = {"format": IR.FORMAT, "label": "HARDWARE", "device": device, "priors": PRIORS,
                           "model": "2 P_L = 1 - A (1 - 2 eps_L)^R, binomial maximum likelihood (dqiqm.stats.fit_eps); "
                                    "A = 1 is the challenge's formula", "bootstrap": 1000, "seed": SEED, "fits": {}}
    sel = {"b1": [1], "b2": [2], "b3": [3], "finetune_b1b2": [1, 2], "test_b3": [3], "all_b1b2b3": [1, 2, 3]}
    plots = {}
    for basis in ("Z", "X"):
        for patch in sorted({p["patch"] for p in pts}):
            for name, batches in sel.items():
                chosen = [p for p in pts if p["basis"] == basis and p["patch"] == patch and p["batch"] in batches]
                if not chosen:
                    continue
                f = fit_both(*pool(chosen))
                fits = f.pop("_fits", None)
                key = f"{basis}_{patch}_{name}"
                out["fits"][key] = {"basis": basis, "patch": patch, "batches": batches, **f}
                if fits and name in ("finetune_b1b2", "test_b3"):
                    plots.setdefault(f"{basis}_{patch}", {})[name] = list(fits)
    sdir = root / "ler" / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / "E1_points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["basis", "patch", "batch", "rounds", "shots", "failures", "raw_failures", "circuit",
                                           "run_id"], lineterminator="\n")
        w.writeheader()
        for p in sorted(pts, key=lambda p: (p["basis"], p["patch"], p["batch"], p["rounds"])):
            w.writerow({k: p[k] for k in w.fieldnames})
    for key, series in plots.items():
        stats.plot_fidelity_decay({f"{k} (A free, A=1)": v for k, v in series.items()},
                                  title=f"HARDWARE ({device}): E1 d=3 {key.replace('_', ', patch ')}, finetune B1+B2 vs test B3",
                                  source=_rel(sdir / "E1_points.csv", root), path=sdir / f"E1_{key}.png")
    out["x_vs_z"] = x_vs_z(out["fits"])
    write_json(sdir / "E1_fits.json", out)
    return out


def x_vs_z(fits: Mapping[str, Any]) -> list[dict[str, Any]]:
    """eps_L(X) / eps_L(Z) per patch and pool, both forms, with the bootstrap intervals side by side."""
    rows = []
    for key, fx in fits.items():
        if not key.startswith("X_"):
            continue
        fzk = "Z_" + key[2:]
        if fzk not in fits or "fixed_A" not in fx or "fixed_A" not in fits[fzk]:
            continue
        fz_ = fits[fzk]
        row = {"pool": key[2:]}
        for form in ("fixed_A", "free_A"):
            ex, ez = fx[form]["eps"], fz_[form]["eps"]
            row[form] = {"eps_X": ex, "eps_X_ci": fx[form]["eps_ci_bootstrap"], "eps_Z": ez, "eps_Z_ci": fz_[form]["eps_ci_bootstrap"],
                         "ratio_X_over_Z": ex / ez if ez else None,
                         "intervals_overlap": not (fx[form]["eps_ci_bootstrap"][0] > fz_[form]["eps_ci_bootstrap"][1]
                                                   or fz_[form]["eps_ci_bootstrap"][0] > fx[form]["eps_ci_bootstrap"][1])}
        rows.append(row)
    return rows


def cmd_summary(args: argparse.Namespace) -> int:
    for device in ("garnet", "emerald"):
        root = DEVICES[device].record
        if not (root / "ler" / "E1").exists():
            continue
        s = summary_fits(root, device)
        for key, f in s["fits"].items():
            if "fixed_A" in f:
                log(f"{device} {key}: eps(A=1) {f['fixed_A']['eps']:.4f} {f['fixed_A']['eps_ci_bootstrap']}, "
                    f"eps(free A) {f['free_A']['eps']:.4f} A {f['free_A']['amplitude']:.3f}")
        for row in s["x_vs_z"]:
            log(f"{device} X/Z {row['pool']}: A=1 {row['fixed_A']['ratio_X_over_Z']:.3f}, free A {row['free_A']['ratio_X_over_Z']:.3f}")
    return 0


def _fmt_ci(ci: Sequence[float] | None, nd: int = 4) -> str:
    return "n/a" if not ci or ci[0] is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def segment_tables() -> list[str]:
    """Per segment: its jobs, ids, calibration sets, sanity and timelines (from ``ler/summary/segments/<task>.json``)."""
    out = []
    for task in ORDER:
        seg = SEGMENTS[task]
        f = DEVICES[seg.device].record / "ler" / "summary" / "segments" / f"{task}.json"
        if not f.exists():
            out += [f"### {task}", "", "Not run yet.", ""]
            continue
        a = json.loads(f.read_text())
        shots = sum(j["total_shots"] for j in a["jobs"])
        out += [f"### {task}: {seg.device}, batch {seg.batch} ({seg.split}), {len(a['jobs'])} jobs, {shots:,} shots", "",
                "| job | job id | calibration set | circuits x shots | ledger (UTC) | queue + compile (s) | execution span (s) | "
                "span per shot (us) | sanity | detection / model |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for j in a["jobs"]:
            tl = j["timeline"]
            ratios = j.get("detection_ratio_to_model") or []
            ratio = f"{min(ratios):.2f}-{max(ratios):.2f}" if ratios else "n/a (no decoder)"
            qc = tl.get("queue_s")
            out.append(f"| {j['label']} | `{j['job_id']}` | `{j['calibration_set_id'][:8]}` | {j['circuits']} x {j['shots']:,} | "
                       f"{j['ledger_time'][11:19]} | {qc if qc is None else f'{qc:.1f}'} | "
                       f"{tl.get('execution_s', float('nan')):.2f} | {tl.get('execution_per_shot_us', float('nan')):.1f} | "
                       f"{j.get('sanity', '')} | {ratio} |")
        out += ["", f"Record problems: {a['record_problems'] or 'none'}. Sanity failures: {a['sanity_failures'] or 'none'}. "
                f"Snapshots: {', '.join(f'`{x}`' for x in a['snapshots'])}.", ""]
        rows = [x for x in a["ler"] if "eps_fixed_A" in x]
        if rows:
            out += ["| LER record | points | shots | eps_L (A = 1) | eps_L (A free) | A |", "|---|---|---|---|---|---|"]
            out += [f"| `{x['experiment']}/{x['run_id']}` | {x['points']} | {x['shots']:,} | {x['eps_fixed_A']:.4f} | "
                    f"{x['eps_free_A']:.4f} | {x['amplitude']:.3f} |" for x in rows]
            out.append("")
    return out


def fit_tables(device: str) -> list[str]:
    f = DEVICES[device].record / "ler" / "summary" / "E1_fits.json"
    if not f.exists():
        return [f"No E1 fits for {device} yet.", ""]
    s = json.loads(f.read_text())
    out = [f"**{device}** (`{_rel(f, REPO)}`; points in `E1_points.csv`; priors {s['priors']}; bootstrap {s['bootstrap']}, "
           f"seed {s['seed']}).", "",
           "| basis | patch | pool | shots | eps_L, A = 1 [bootstrap 95%] | eps_L, A free [bootstrap 95%] | A [bootstrap 95%] | "
           "chi2/dof (A = 1, A free) |", "|---|---|---|---|---|---|---|---|"]
    for key, x in s["fits"].items():
        if "fixed_A" not in x:
            continue
        fa, fr = x["fixed_A"], x["free_A"]
        out.append(f"| {x['basis']} | {x['patch']} | {key.split('_', 2)[2]} | {sum(x['shots']):,} | {fa['eps']:.4f} "
                   f"{_fmt_ci(fa['eps_ci_bootstrap'])} | {fr['eps']:.4f} {_fmt_ci(fr['eps_ci_bootstrap'])} | {fr['amplitude']:.3f} "
                   f"{_fmt_ci(fr['amplitude_ci_bootstrap'], 3)} | {fa['chi2_per_dof']:.2f}, {fr['chi2_per_dof']:.2f} |")
    out += ["", "| pool | eps_X / eps_Z, A = 1 | intervals overlap | eps_X / eps_Z, A free | intervals overlap |", "|---|---|---|---|---|"]
    for r in s["x_vs_z"]:
        out.append(f"| {r['pool']} | {r['fixed_A']['ratio_X_over_Z']:.3f} | {r['fixed_A']['intervals_overlap']} | "
                   f"{r['free_A']['ratio_X_over_Z']:.3f} | {r['free_A']['intervals_overlap']} |")
    return out + [""]


def p_l_tables(device: str) -> list[str]:
    """P_L(R) with Wilson 95% per (basis, patch, batch): the points every fit is made from."""
    f = DEVICES[device].record / "ler" / "summary" / "E1_fits.json"
    if not f.exists():
        return []
    s = json.loads(f.read_text())
    out = []
    for key, x in s["fits"].items():
        if key.split("_", 2)[2] not in ("b1", "b2", "b3"):
            continue
        cells = " ".join(f"R={r}: {k}/{n} = {k / n:.4f} {_fmt_ci(ci)};" for r, k, n, ci in
                         zip(x["rounds"], x["failures"], x["shots"], x["wilson95"]))
        out.append(f"- {device} {x['basis']} {x['patch']} {key.split('_', 2)[2]}: {cells}")
    return out + [""]


def e3_table() -> list[str]:
    f = DEVICES["garnet"].record / "derived" / "E3" / "b1_pooled" / "summary.json"
    if not f.exists():
        return ["Not written yet.", ""]
    s = json.loads(f.read_text())
    fmt = lambda v: "" if v is None else f"{v:.4f}"  # noqa: E731
    out = [f"`{_rel(f, REPO)}` and `readout_features.csv` beside it ({len(s['runs'])} E3 runs of batch 1). {s['flips']}.", "",
           "| qubit | P(1\\|0) | P(0\\|1) | record flip, prep 0 | record flip, prep 1 | state flip, prep 0 | state flip, prep 1 |",
           "|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        out.append(f"| {r['qb']} | {fmt(r['p1_given_0'])} | {fmt(r['p0_given_1'])} | {fmt(r['record_flip_prep0'])} | "
                   f"{fmt(r['record_flip_prep1'])} | {fmt(r['state_flip_prep0'])} | {fmt(r['state_flip_prep1'])} |")
    med = s["median"]
    out += ["", "Medians: " + ", ".join(f"{k} {fmt(v)}" for k, v in med.items()) + ".", ""]
    return out


def cmd_doc(args: argparse.Namespace) -> int:
    notes = Path(args.notes).read_text() if args.notes else ""
    lines = [
        "# The Garnet and Emerald batches", "",
        "The main hardware dataset: 6 segments, 26 jobs, 1,419,000 shots on IQM Garnet and IQM Emerald, run back to back "
        "in the brief's order (`garnet-b1`, `garnet-b2`, `emerald-b1`, `emerald-b2`, `garnet-b3`, `emerald-b3`), so each "
        "device's test batch (B3) is time-separated from its own training batches by the other device's work. "
        "Driver: `scripts/batch_run.py`. Generated by `scripts/batch_run.py doc`; the notes come from the handoff.", "",
        notes, "",
        "## The segments", "",
        *segment_tables(),
        "## The LER fits (E1)", "",
        "2 P_L = 1 - A (1 - 2 eps_L)^R, binomial maximum likelihood, both forms: the challenge's A = 1 and A free. "
        "Pools: per batch, finetune (B1 + B2) and test (B3), and all three.", "",
        *fit_tables("garnet"), *fit_tables("emerald"),
        "### P_L(R) per batch, with Wilson 95%", "",
        *p_l_tables("garnet"), *p_l_tables("emerald"),
        "## E3: Garnet's readout features (batch 1)", "",
        *e3_table(),
        "## How to reproduce", "",
        "```",
        "uv sync --locked && uv run pytest",
        "uv run python scripts/batch_run.py plan",
        "uv run python scripts/batch_run.py verify --segment garnet-b1        # offline freeze + references",
        "flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\",
        "    uv run python scripts/batch_run.py analyze --segment <task> --root <scratch> --snapshots <before> <after>",
        "uv run python scripts/batch_run.py summary && uv run python scripts/batch_run.py doc --notes <notes.md>",
        "```", "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    log(f"wrote {_rel(DOC, REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="offline: the brief's table against both job plans")
    s = sub.add_parser("snapshot", help="live, read-only: resonance_check.py")
    s.add_argument("--segment", choices=ORDER, required=True)
    v = sub.add_parser("verify", help="offline freeze; with --snapshot the live checks; with --live the read-only requests")
    v.add_argument("--segment", choices=ORDER, required=True)
    v.add_argument("--snapshot", type=Path)
    v.add_argument("--live", action="store_true")
    v.add_argument("--rerank", action="store_true", help="re-rank the placements under a changed calibration (information)")
    v.add_argument("--out", type=Path)
    s = sub.add_parser("submit", help="the segment's jobs (armed by DQ_QPU=run ...)")
    s.add_argument("--segment", choices=ORDER, required=True)
    s.add_argument("--snapshot", type=Path, required=True, help="the before snapshot (its calibration set is used)")
    s.add_argument("--timeout", type=float, default=None, help="seconds to wait for each result (default: iqm-client's)")
    r = sub.add_parser("recover", help="read-only: re-pull a submitted job and save it as save_raw does")
    r.add_argument("--segment", choices=ORDER, required=True)
    r.add_argument("--job-id", required=True)
    r.add_argument("--snapshot", type=Path, help="a snapshot of the job's calibration set, for the sanity references")
    r.add_argument("--timeout", type=float, default=None)
    a = sub.add_parser("analyze", help="offline: the records")
    a.add_argument("--segment", choices=ORDER, required=True)
    a.add_argument("--snapshots", type=Path, nargs="+", required=True, help="before (first), any mid-segment, after (last)")
    a.add_argument("--root", type=Path, default=None, help="the record root (default: the device's InfoRecord)")
    a.add_argument("--ledger", type=Path, default=Path(hw.DEFAULT_LEDGER))
    a.add_argument("--out", type=Path, default=None)
    e = sub.add_parser("e3", help="offline: E3's readout features, pooled per qubit")
    e.add_argument("--segment", choices=ORDER, required=True)
    sub.add_parser("summary", help="offline: pooled LER fits and X against Z")
    d = sub.add_parser("doc", help="offline: write docs/batches.md from the records")
    d.add_argument("--notes", type=Path, default=None, help="a markdown file of notes for the top of the doc")
    args = ap.parse_args()
    return {"plan": cmd_plan, "snapshot": cmd_snapshot, "verify": cmd_verify, "submit": cmd_submit, "recover": cmd_recover,
            "analyze": cmd_analyze, "e3": cmd_e3, "summary": cmd_summary, "doc": cmd_doc}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
