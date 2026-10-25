"""emerald-dd: dynamical decoupling on and off on IQM Emerald, a paired A/B. 2 segments, 7 jobs, 81,000 shots.

    uv run python scripts/emerald_dd.py plan                                    # offline: the allowance, the jobs, the arming
    uv run python scripts/emerald_dd.py predict                                 # offline (sim lock): structure + Stim's prediction
    uv run python scripts/emerald_dd.py snapshot                                # live, read-only: one Emerald snapshot
    uv run python scripts/emerald_dd.py verify --segment S [--snapshot P [--live]]
    DQ_QPU=run DQ_QPU_TASK=S DQ_QPU_MAX_JOBS=<jobs> DQ_QPU_MAX_SHOTS=<shots> \\
        uv run python scripts/emerald_dd.py submit --segment S --snapshot P     # the segment's jobs, in order
    uv run python scripts/emerald_dd.py recover --segment S --job-id J          # read-only; only if submit failed after a job id
    uv run python scripts/emerald_dd.py analyze --snapshots P Q ...             # offline (sim lock): the records and the A/B
    uv run python scripts/emerald_dd.py stabilizers                             # offline: per stabilizer, on against off
    uv run python scripts/emerald_dd.py doc                                     # offline: docs/emerald-dd.md's tables

The design is ``docs/emerald-design.md``, "Dynamical decoupling: an A/B to run on E1". IQM's server fills a
circuit's idle windows with DD pulse sequences when the job's ``CircuitCompilationOptions`` say
``dd_mode=ENABLED``. DD is the identity, so the circuits, their detectors and their decoders are unchanged.
The options belong to a job, so an A/B is a pair of jobs on the same circuits.

The segments (the brief's QPU allowance, ``SEGMENTS``, in this order)
    ``emerald-dd-pilot``: 1 job, ``DDP``: ``E1_d3_X_r03_eA``, DD on, 1,000 shots.
    ``emerald-dd``: the E1 ABBA, 4 jobs ``A1`` (on), ``B1`` (off), ``B2`` (off), ``A2`` (on), each E1 Z and X at
    R in {3, 8, 15} on eA (6 circuits x 2,000 shots), back to back, so a linear drift cancels in A1+A2 against
    B1+B2; then E4 at 5 us, ``E4A`` (on) and ``E4B`` (off), each the 8 frozen 5 us circuits on eA (Z and X at
    R in {2, 3, 5, 8}) x 2,000 shots. 6 jobs, 80,000 shots: ``DQ_QPU_MAX_SHOTS=80000`` (the brief allows
    96,000; E4 has 8 frozen 5 us circuits, not 12). Every job of a segment runs its circuits in one fixed order.

The options
    DD on is ``CircuitCompilationOptions(dd_mode=DDMode.ENABLED, dd_strategy=STANDARD_DD_STRATEGY)``: the
    strategy is named (``docs/hw.md``), so ``meta.json`` holds the sequences that ran (EDD ``XYXYYXYX`` in a
    window of at least 9 PRX durations, XY4 ``YXYX`` from 5, CPMG ``XX`` from 2). DD off is the defaults
    (``None``). ``hw.submit`` records every field in ``meta.json``'s ``compilation_options`` and tags the
    ledger purpose ``[dd=ENABLED,dd_strategy=879fe63c]``. After each job, the server's own record of it is
    re-pulled read-only (``hw.retrieve_native(job_id).parameters``, the job's payload): its ``dd_mode`` and
    strategy must be the job's, and its circuits must hash to the frozen ``native_sha256``.

The freeze (``scripts/batch_run.py``'s, branch batches ``47eb36a``, copied for Emerald alone)
    Every circuit is rebuilt from the committed snapshot the plan was made from,
    ``emerald.build(emerald.live_calibration(emerald.DEFAULT_SNAPSHOT))`` (calibration set 23e12f6b), and its
    ``native_sha256``, ``stim_sha256`` and ``qasm_sha256`` must equal ``InfoRecord/emerald/circuits/<exp>/
    placement.json``'s. A live calibration never rebuilds a circuit. Against a live snapshot, each circuit's
    qubits must still serve ``prx`` and ``measure`` and its ``cz`` loci must still be calibrated
    (``emerald.check_live_couplers``); a lost qubit or coupler stops the segment. Before each job, the very
    request ``hw.submit`` will build, options included, is built on the read-only backend and checked circuit
    by circuit (instructions = ``native_sha256``, ``hw._sent_qubits`` = the placement's qubits, eA's 17).
    ``hw.submit`` also gets ``circuit_names`` and ``native_sha256``, so it refuses a request that differs.

Submitting
    ``submit`` refuses unless armed exactly as the segment's row, against the default ledger; ``emerald-dd``
    refuses unless the pilot is in the ledger and passed its checks. Before each job it re-reads the server's
    default calibration set (a new snapshot and the live checks if it moved). It writes the job's manifest
    (``raw/<YYYYMMDD>-<task>/<label>.json``: position -> circuit, ``native_sha256``, qubits, DD state) before
    ``hw.submit``, prints the job id the moment the ledger row lands, then checks the row, ``meta.json``, the
    server's record and every circuit's sanity. A failed check stops the segment. A job in the ledger is never
    submitted again; one whose raw data is missing is recovered read-only (``recover``).

The sanity checks (the batch's, per circuit)
    the measurement shape; all-zero rows; the first readout of every ancilla, from a noiseless sample of the
    circuit's Stim source (X basis: the X-type ancillas deterministic, the Z-type random; Z the reverse); the
    detection fraction well below 0.5; ``meta.json``'s ``physical_qubits``. The pilot also needs: the server's
    ``dd_mode`` enabled, the timeline's execution span, the detection fraction within [0.5, 2] x the decoder
    priors' own (``hardware_noise``), and the decoded P_L at most twice the batch's DD-off P_L of the same
    circuit (``BATCH_PILOT_REF``).

The records
    ``analyze`` writes every run with the record's writers (``datasets/``, ``derived/``, ``ler/``), labelled
    HARDWARE. Run ids are ``dd_<job>_<on|off>_<circuit>``. The DD state is in each manifest's ``raw`` block
    (``dd``, ``compilation_options``, the job label): ``inforecord.MANIFEST`` has no ``compilation`` key yet,
    and ``check_run`` refuses a key outside it. The shots are not in the ML split: their ``split`` is
    ``dd_ab``, never finetune, validation or test. The record's decoder (``pm_fail``) is the batch's,
    PyMatching on ``device.hardware_noise(cal, spec)`` (Ramsey T2); ``derived/.../dd_checks.json`` also has
    the echo-T2 variant (``hardware_noise(..., t2="echo")``) for every run. Both are reported; neither is
    chosen on these shots.

The analysis (``ler/E1/dd_ab/ab.json``, ``ler/E4/dd_ab/ab.json``)
    P_L per basis and R, DD on (A1+A2) against off (B1+B2), Wilson intervals, the difference with Newcombe's
    interval, and the drift A1 - A2 and B1 - B2 beside it; the same for the echo decoder and for the
    undecoded observable flips; the detection fraction per round; eps_L per arm (and the difference, by a
    seeded parametric bootstrap); E4 at 5 us, E4A against E4B. ``stabilizers`` breaks the detection rate down per
    stabilizer (``by_stabilizer.json``), beside the batches' DD-off rate of the same circuit, the priors' own model
    and the data qubits' frame through the readout window, and gives each deterministic first readout's misread rate.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
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

REPO = Path(__file__).resolve().parents[1]
RECORD = EM.RECORD
RAW_ROOT = RECORD / "raw"
CALIBRATION = REPO / "data" / "calibration"
DOC = REPO / "docs" / "emerald-dd.md"
PROGRESS = Path("/home/sal/duquantumIQM/orchestration/reports/emerald-dd-progress.md")
DEVICE = EM.DEVICE

NATIVE_OPS = frozenset({"prx", "cz", "measure", "barrier", "delay"})
#: Patch eA: QB13, QB14, QB19-QB22, QB27-QB31, QB36-QB39, QB44, QB45.
PATCH_QUBITS = tuple(f"QB{i}" for i in (13, 14, *range(19, 23), *range(27, 32), *range(36, 40), 44, 45))
#: Noiseless samples that tell a deterministic first readout from a random one, the model's shots, the seed.
REF_SHOTS, MODEL_SHOTS, SEED = 1024, 20_000, 2026
#: The batch's sanity bands (``scripts/batch_run.py``).
RANDOM_BAND = (0.4, 0.6)
DETERMINISTIC_MAX = 0.3
DETECTION_MAX = 0.45
DETECTION_HIGH_MODEL, DETECTION_MAX_HIGH = 0.3, 0.48
ALL_ZERO_PER_1000 = 5
#: The pilot's extra bands: detection fraction against the priors' own, and P_L against the batch's DD-off P_L.
PILOT_DETECTION_RATIO = (0.5, 2.0)
PILOT_P_L_RATIO_MAX = 2.0
#: The batch's DD-off hardware numbers for the pilot's circuit, ``E1_d3_X_r03_eA`` (branch batches 47eb36a,
#: ``InfoRecord/emerald/derived/E1/b<k>_E1_d3_X_r03_eA/batch_checks.json``; decoder ``hardware_noise``).
BATCH_PILOT_REF = (
    {"batch": 1, "job_id": "01a17093-aba4-773f-8755-4727f0c943ee", "detection_fraction": 0.134208, "pm_failures": 167,
     "obs_flips": 494, "shots": 2000},
    {"batch": 2, "job_id": "01a17097-6a9d-70e2-9ec4-6b8669dce3d4", "detection_fraction": 0.136833, "pm_failures": 199,
     "obs_flips": 518, "shots": 2000},
    {"batch": 3, "job_id": "01a170a1-17bf-77ad-a986-fadac8637f39", "detection_fraction": 0.124729, "pm_failures": 167,
     "obs_flips": 442, "shots": 2000},
)
#: The decoder priors: the batch's (Ramsey T2), and the echo-T2 variant the design names for DD shots.
PRIORS = {"ramsey": "hardware_noise (79092cf)", "echo": "hardware_noise t2='echo' (79092cf)"}
SPLIT_LABEL = "dd_ab"
#: The parametric bootstrap of the eps difference.
EPS_BOOTSTRAP = 1000


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


def _plain(v: Any) -> Any:
    """``v`` as JSON would give it back: tuples as lists, enums as their values."""
    return json.loads(json.dumps(v, default=lambda x: getattr(x, "value", str(x))))


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
# The allowance: the brief's table.
# ---------------------------------------------------------------------------

PILOT_CIRCUIT = "E1_d3_X_r03_eA"
#: E1 Z and X at R in {3, 8, 15} on eA, one fixed order for every ABBA job.
E1_CIRCUITS = tuple(f"E1_d3_{b}_r{r:02d}_eA" for r in (3, 8, 15) for b in ("Z", "X"))
#: The frozen E4 circuits at 5 us on eA: Z and X at R in {2, 3, 5, 8}.
E4_CIRCUITS = tuple(f"E4_d3_{b}_r{r:02d}_eA_t05000ns" for r in (2, 3, 5, 8) for b in ("Z", "X"))


@dataclass(frozen=True)
class DDJob:
    label: str
    dd: bool
    experiment: str
    names: tuple[str, ...]
    shots: int
    position: int  # in the segment, 0-based

    @property
    def total_shots(self) -> int:
        return self.shots * len(self.names)

    @property
    def arm(self) -> str:
        return "on" if self.dd else "off"


@dataclass(frozen=True)
class Segment:
    task: str
    jobs: tuple[DDJob, ...]
    max_jobs: int  # the brief's allowance
    max_shots: int

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    @property
    def total_shots(self) -> int:
        return sum(j.total_shots for j in self.jobs)

    def arming(self) -> dict[str, str]:
        """Exactly the jobs and shots planned: the brief sets ``DQ_QPU_MAX_SHOTS`` to the exact sum."""
        return {"DQ_QPU": "run", "DQ_QPU_TASK": self.task, "DQ_QPU_MAX_JOBS": str(self.n_jobs),
                "DQ_QPU_MAX_SHOTS": str(self.total_shots)}

    def job(self, label: str) -> DDJob:
        return next(j for j in self.jobs if j.label == label)


SEGMENTS: dict[str, Segment] = {s.task: s for s in (
    Segment("emerald-dd-pilot", (DDJob("DDP", True, "E1", (PILOT_CIRCUIT,), 1000, 0),), 1, 1000),
    Segment("emerald-dd", (
        DDJob("A1", True, "E1", E1_CIRCUITS, 2000, 0),
        DDJob("B1", False, "E1", E1_CIRCUITS, 2000, 1),
        DDJob("B2", False, "E1", E1_CIRCUITS, 2000, 2),
        DDJob("A2", True, "E1", E1_CIRCUITS, 2000, 3),
        DDJob("E4A", True, "E4", E4_CIRCUITS, 2000, 4),
        DDJob("E4B", False, "E4", E4_CIRCUITS, 2000, 5),
    ), 6, 96_000),
)}
ORDER = tuple(SEGMENTS)


def check_segments(placement: Mapping[str, Any] | None = None) -> list[str]:
    """The segments against the brief: jobs, shots, the ABBA order, eA only. Returns the problems (none: [])."""
    problems = []
    for seg in SEGMENTS.values():
        if seg.n_jobs != seg.max_jobs:
            problems.append(f"{seg.task}: {seg.n_jobs} jobs planned, the brief grants {seg.max_jobs}")
        if seg.total_shots > seg.max_shots:
            problems.append(f"{seg.task}: {seg.total_shots:,} shots planned > the brief's {seg.max_shots:,}")
        if [j.position for j in seg.jobs] != list(range(seg.n_jobs)) or len({j.label for j in seg.jobs}) != seg.n_jobs:
            problems.append(f"{seg.task}: job positions or labels")
        for j in seg.jobs:
            if len(set(j.names)) != len(j.names):
                problems.append(f"{seg.task} {j.label}: a repeated circuit")
            if not all(n.startswith(j.experiment + "_") and "_eA" in n for n in j.names):
                problems.append(f"{seg.task} {j.label}: a circuit not of {j.experiment} on eA")
            if placement is not None:
                for n in j.names:
                    if n not in placement:
                        problems.append(f"{seg.task} {j.label}: {n} is not a frozen circuit")
                    elif set(placement[n]["placement"].values()) - set(PATCH_QUBITS):
                        problems.append(f"{seg.task} {j.label}: {n} is not on eA")
    pilot, main = SEGMENTS["emerald-dd-pilot"], SEGMENTS["emerald-dd"]
    if [(j.label, j.dd, j.names, j.shots) for j in pilot.jobs] != [("DDP", True, (PILOT_CIRCUIT,), 1000)]:
        problems.append("emerald-dd-pilot: not 1 job of E1 X R=3 on eA, DD on, 1,000 shots")
    e1 = [j for j in main.jobs if j.experiment == "E1"]
    if [j.dd for j in e1] != [True, False, False, True] or any(j.names != E1_CIRCUITS or j.shots != 2000 for j in e1):
        problems.append("emerald-dd: the E1 jobs are not ABBA (on, off, off, on) of the same 6 circuits x 2,000")
    e4 = [j for j in main.jobs if j.experiment == "E4"]
    if [j.dd for j in e4] != [True, False] or any(j.names != E4_CIRCUITS or len(j.names) > 12 for j in e4):
        problems.append("emerald-dd: the E4 jobs are not (on, off) of the same 5 us circuits")
    if [j.experiment for j in main.jobs] != ["E1"] * 4 + ["E4"] * 2:
        problems.append("emerald-dd: E1's ABBA first, then E4")
    return problems


def compilation_options(dd: bool) -> Any:
    """The job's ``CircuitCompilationOptions``: DD enabled with the named standard strategy, or the defaults (None)."""
    if not dd:
        return None
    from iqm.iqm_client.models import STANDARD_DD_STRATEGY, CircuitCompilationOptions, DDMode

    return CircuitCompilationOptions(dd_mode=DDMode.ENABLED, dd_strategy=STANDARD_DD_STRATEGY)


def options_record(dd: bool) -> dict[str, Any]:
    """What ``hw.submit`` writes to ``meta.json``'s ``compilation_options`` for a DD-on or DD-off job."""
    return hw.compilation_options_record(compilation_options(dd))


def purpose(seg: Segment, job: DDJob) -> str:
    """The purpose given to ``hw.submit``: ``<task> <label> ...`` first, so the label parses back out of the row."""
    return f"{seg.task} {job.label} {job.experiment} DD {job.arm} ({len(job.names)} circuits x {job.shots} shots)"


def tagged_purpose(seg: Segment, job: DDJob) -> str:
    """The purpose as the ledger row and ``meta.json`` carry it: ``hw.submit`` appends the options' tag."""
    return hw._tagged(purpose(seg, job), hw.compilation_options_tag(options_record(job.dd)))  # noqa: SLF001


# ---------------------------------------------------------------------------
# The frozen circuits (``scripts/batch_run.py``'s ``Frozen``, Emerald only).
# ---------------------------------------------------------------------------


def load_placement() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted((RECORD / "circuits").glob("*/placement.json")):
        out.update(json.loads(p.read_text()))
    return out


class Frozen:
    """Emerald's circuits rebuilt from the committed snapshot the plan was made from, never from a live one."""

    def __init__(self) -> None:
        self.cal = EM.live_calibration(EM.DEFAULT_SNAPSHOT)
        if self.cal.calibration_set_id != EM.CALSET:
            raise SystemExit(f"FREEZE: the committed snapshot is {self.cal.calibration_set_id}, not {EM.CALSET}")
        self.suite = EM.build(self.cal)
        self.placement = load_placement()
        problems = check_segments(self.placement)
        if problems:
            raise SystemExit("the segments disagree with the brief: " + "; ".join(problems))

    def planned_qubits(self, name: str) -> list[str]:
        return sorted(set(self.placement[name]["placement"].values()), key=dev.qubit_sort_key)

    def names_map(self, job: DDJob) -> dict[int, str]:
        return merged_names([self.suite[n].hardware.qubit_index_to_name for n in job.names])

    def circuits(self, job: DDJob) -> list[Any]:
        return [self.suite[n].hardware.circuit for n in job.names]

    def hashes(self, job: DDJob) -> list[str]:
        return [self.placement[n]["native_sha256"] for n in job.names]

    def mappings(self, job: DDJob) -> list[dict[int, str]]:
        """What ``hw.submit`` writes to ``meta.json``'s ``qubit_mapping`` for this job."""
        return hw._qubit_mappings(None, self.circuits(job), self.names_map(job))  # noqa: SLF001

    def measured(self, job: DDJob) -> list[dict[str, str]]:
        """Per circuit: memory column j -> the device qubit Stim's measurement j is on (what ``measured_qubits`` must be)."""
        out = []
        for n in job.names:
            spec = self.suite[n]
            meas_q = C.measured_qubits(spec.stim)
            out.append({str(j): spec.physical[q] for j, q in enumerate(meas_q)})
        return out


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


def freeze_checks(fz: Frozen, job: DDJob) -> list[dict[str, Any]]:
    """Every circuit of a job, rebuilt from the committed snapshot, against ``placement.json``. Raises ``AssertionError``."""
    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey

    arch_couplers = {frozenset(c) for c in fz.suite.arch.couplers}
    out = []
    for name in job.names:
        spec, entry = fz.suite[name], fz.placement[name]
        hc = spec.hardware
        assert spec.experiment == job.experiment, (name, spec.experiment)
        assert hc.target == "hardware" and entry["target"] == "hardware", name
        assert entry["device"] == DEVICE and spec.params["patch"] == "eA", (name, entry["device"], spec.params)
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
        EM.check_live_couplers(spec, fz.cal)
        qubits = sorted({q for i in sent for q in i.locus}, key=dev.qubit_sort_key)
        assert qubits == fz.planned_qubits(name) == list(PATCH_QUBITS), f"{name}: sent to {qubits}, placed on {fz.planned_qubits(name)}"
        n = spec.stim.num_measurements
        meas_q = C.measured_qubits(spec.stim)
        clbits = []
        for inst in (i for i in sent if i.name == "measure"):
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


def load_snapshot(path: Path) -> dict[str, Any]:
    snap = json.loads(Path(path).read_text())
    if snap.get("device") != DEVICE:
        raise SystemExit(f"{path}: device {snap.get('device')!r}, not {DEVICE}")
    return snap


def live_checks(fz: Frozen, jobs: Sequence[DDJob], snapshot: Path) -> dict[str, Any]:
    """The frozen circuits against a live snapshot's calibration: nothing rebuilt. Raises ``AssertionError`` on a lost gate."""
    snap = load_snapshot(snapshot)
    arch = snap["architecture"]
    live_cal = EM.live_calibration(snapshot)
    live = {frozenset(k) for k in live_cal.couplers}
    lost = {}
    for name in sorted({n for job in jobs for n in job.names}):
        spec = fz.suite[name]
        qubits, cz = circuit_gates(spec)
        missing = {
            "prx": [q for q in qubits if not _served(arch, "prx", (q,))],
            "measure": [q for q in qubits if not _served(arch, "measure", (q,))],
            "cz": [f"{a}-{b}" for a, b in cz if not _served(arch, "cz", (a, b)) or frozenset((a, b)) not in live],
        }
        try:
            EM.check_live_couplers(spec, live_cal)
        except AssertionError as e:
            missing["check_live_couplers"] = [str(e)]
        if any(missing.values()):
            lost[name] = missing
    assert not lost, f"STOP: the live calibration lost a gate a frozen circuit needs: {lost}"
    return {"snapshot": _rel(snapshot, REPO), "calibration_set_id": snap["calibration_set_id"],
            "same_as_design": snap["calibration_set_id"] == EM.CALSET,
            "circuits": len({n for job in jobs for n in job.names}), "lost": {}}


def instructions_sha256(instructions: Sequence[Any]) -> str:
    """``circuits.native_json``'s hash for any sequence of serialized instructions (a run request's too)."""
    return hw.instructions_sha256(instructions)


def run_request_checks(rr: Any, fz: Frozen, job: DDJob, calset: str) -> dict[str, Any]:
    """The run request ``hw.submit`` would build for a job, circuit by circuit and option by option. Raises ``AssertionError``."""
    from iqm.iqm_client.models import STANDARD_DD_STRATEGY, DDMode, HeraldingMode

    assert len(rr.circuits) == len(job.names), (len(rr.circuits), len(job.names))
    sent = hw._sent_qubits(rr)  # noqa: SLF001 -- what the ledger row will record
    for i, name in enumerate(job.names):
        sha = instructions_sha256(rr.circuits[i].instructions)
        assert sha == fz.placement[name]["native_sha256"], f"position {i}, {name}: the run request's instructions hash to {sha}"
        assert sent[i] == fz.planned_qubits(name), f"position {i}, {name}: sent to {sent[i]}"
    assert rr.qubit_mapping is None, rr.qubit_mapping
    assert str(rr.calibration_set_id) == calset, rr.calibration_set_id
    assert rr.shots == job.shots, rr.shots
    want = options_record(job.dd)
    got = {k: hw._option_value(getattr(rr, k)) for k in want}  # noqa: SLF001 -- the record's own rule
    assert _plain(got) == _plain(want), f"{job.label}: the request's options {got} != {want}"
    assert rr.dd_mode == (DDMode.ENABLED if job.dd else DDMode.DISABLED), rr.dd_mode
    assert rr.dd_strategy == (STANDARD_DD_STRATEGY if job.dd else None), rr.dd_strategy
    assert rr.heralding_mode == HeraldingMode.NONE and rr.active_reset_cycles is None
    return {"label": job.label, "dd": job.arm, "circuits": len(rr.circuits), "shots": rr.shots,
            "calibration_set_id": str(rr.calibration_set_id),
            "qubits": sorted({q for s in sent for q in s}, key=dev.qubit_sort_key),
            "instructions_sha256_equal": True, "qubit_mapping": None, "options": _plain(got)}


def live_run_request(fz: Frozen, job: DDJob, calset: str) -> Any:
    """``create_run_request`` exactly as ``hw.submit`` calls it, options included, on the read-only backend."""
    b = hw.backend(DEVICE, calset)
    assert str(b.architecture.calibration_set_id) == calset, b.architecture.calibration_set_id
    return b.create_run_request(fz.circuits(job), shots=job.shots, qubit_index_to_name=fz.names_map(job),
                                circuit_compilation_options=compilation_options(job.dd))


# ---------------------------------------------------------------------------
# The decoder priors, and the sanity references (``scripts/batch_run.py``'s).
# ---------------------------------------------------------------------------


def decoder_priors(cal: dev.Calibration, spec: Any, t2: str = "ramsey") -> tuple[Any, dict[str, Any], str]:
    """(noisy copy, ``hardware_noise``'s info, note): the batch's priors, or with ``t2="echo"`` the echo-T2 variant."""
    nm, info = dev.hardware_noise(cal, spec, t2=t2)
    note = (f"priors: {PRIORS[t2]}. dqiqm.experiments.device.hardware_noise(calibration set {cal.calibration_set_id}, "
            f"spec{', t2=echo' if t2 == 'echo' else ''}): calibrated_noise with the {info['idle_channel']} idle over "
            f"{info['idle_ns']:g} ns per mid-circuit readout layer (readout window {info['readout_window_ns']:g} ns from "
            f"{info['readout_window_source']}, plus a {info['delay_ns']:g} ns delay), T2 {info['t2']}, final readout "
            f"{info['final_readout']}")
    return spec.noisy(nm), info, note


def first_readouts(spec: Any) -> list[int]:
    """The column of each non-data qubit's first measurement (round 1's ancilla readout)."""
    meas_q = C.measured_qubits(spec.stim)
    cols = {}
    for j, q in enumerate(meas_q):
        if spec.roles.get(q, "data") != "data":
            cols.setdefault(q, j)
    return sorted(cols.values())


def reference(spec: Any, cal: dev.Calibration | None, *, model: bool = True) -> dict[str, Any]:
    """What a sane run of ``spec`` looks like: the first readouts' pattern from a noiseless sample, and the model's."""
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
    meas_q = C.measured_qubits(spec.stim)
    for j in cols:
        st = spec.stabilizers.get(meas_q[j])
        if st is not None:
            out["x_type_first_cols" if st.pauli == "X" else "z_type_first_cols"].append(j)
    out["model"] = None
    if model and cal is not None:
        noisy, info, _ = decoder_priors(cal, spec)
        raw = noisy.compile_sampler(seed=SEED).sample(MODEL_SHOTS).astype(np.uint8)
        dets, _ = D.detection_events(spec.stim, raw)
        out["model"] = {"shots": MODEL_SHOTS, "seed": SEED, "idle_ns": info["idle_ns"], "detection_fraction": float(dets.mean()),
                        "all_zero": float((~raw.any(axis=1)).mean()),
                        "random_first_mean": float(raw[:, random_cols].mean()) if random_cols else None,
                        "deterministic_first_misread": float((raw[:, det_cols] != ref[det_cols]).mean()) if det_cols else None}
    return out


def sanity(raw: np.ndarray, dets: np.ndarray, ref: Mapping[str, Any], *, shots: int,
           physical: Any = None, planned: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The batch's sanity checks on one circuit's record: each with its value, band and pass."""
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


def pilot_checks(raw: np.ndarray, dets: np.ndarray, obs: np.ndarray, ref: Mapping[str, Any], *, pm_failures: int,
                 timeline: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The pilot's checks beyond the batch's: plausible against the simulation and against the batch's DD-off run."""
    checks = []
    statuses = {str(e.get("status")) for e in timeline or [] if isinstance(e, Mapping)}
    checks.append({"check": "timeline: execution_started and execution_ended", "value": sorted(statuses),
                   "band": ["execution_started", "execution_ended"],
                   "pass": {"execution_started", "execution_ended"} <= statuses})
    model = (ref.get("model") or {}).get("detection_fraction")
    f = float(np.asarray(dets).mean())
    lo, hi = (PILOT_DETECTION_RATIO[0] * model, PILOT_DETECTION_RATIO[1] * model) if model else (None, None)
    checks.append({"check": "detection fraction against the priors' own", "value": f, "band": [lo, hi],
                   "pass": model is not None and lo <= f <= hi,
                   "info": {"model": model, "ratio": f / model if model else None,
                            "batch_dd_off": [r["detection_fraction"] for r in BATCH_PILOT_REF]}})
    n = len(raw)
    p = pm_failures / n
    ref_p = max(r["pm_failures"] / r["shots"] for r in BATCH_PILOT_REF)
    checks.append({"check": "decoded P_L against the batch's DD-off P_L", "value": p,
                   "band": f"<= {PILOT_P_L_RATIO_MAX:g} x {ref_p:.4f} = {PILOT_P_L_RATIO_MAX * ref_p:.4f}",
                   "pass": p <= PILOT_P_L_RATIO_MAX * ref_p,
                   "info": {"pm_failures": pm_failures, "shots": n, "wilson95": list(stats.wilson(pm_failures, n)),
                            "obs_flips": int(np.asarray(obs)[:, 0].sum()),
                            "batch_dd_off": [{k: r[k] for k in ("batch", "job_id", "pm_failures", "shots")}
                                             for r in BATCH_PILOT_REF]}})
    return checks


# ---------------------------------------------------------------------------
# The ledger, the manifests, the guards.
# ---------------------------------------------------------------------------


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


def find_job_dir(raw_root: Path, task: str, job_id: str) -> Path | None:
    """The job's raw directory, whatever day it landed on (``raw/<YYYYMMDD>-<task>/<job id>/``)."""
    dirs = [p.parent for p in Path(raw_root).glob(f"*-{task}/{job_id}/meta.json")]
    if len(dirs) > 1:
        raise SystemExit(f"job {job_id} is saved twice: {dirs}")
    return dirs[0] if dirs else None


def manifest_path(raw_root: Path, seg: Segment, job: DDJob, day: str) -> Path:
    return Path(raw_root) / f"{day}-{seg.task}" / f"{job.label}.json"


def checked_ok(raw_root: Path, seg: Segment, job: DDJob, ledger: Path) -> bool | None:
    """A ledger job's manifest verdict: True (checked OK), False (failed), None (not in the ledger, or never checked)."""
    rows = job_rows(ledger, seg).get(job.label, [])
    if len(rows) != 1:
        return None
    mpath = manifest_path(raw_root, seg, job, rows[0].time[:10].replace("-", ""))
    if not mpath.exists():
        return None
    return json.loads(mpath.read_text()).get("ok")


def check_order(seg: Segment, ledger: Path, raw_root: Path) -> None:
    """Refuse unless every earlier segment ran and passed its checks, and no later one has started."""
    i = ORDER.index(seg.task)
    for task in ORDER[:i]:
        prev = SEGMENTS[task]
        n = len(ledger_rows(ledger, task))
        if n != prev.n_jobs:
            raise SystemExit(f"refusing: {task} comes before {seg.task} and has {n} of {prev.n_jobs} jobs in the ledger")
        for job in prev.jobs:
            if checked_ok(raw_root, prev, job, ledger) is not True:
                raise SystemExit(f"refusing: {task} {job.label} did not pass its checks; {seg.task} runs only after it does")
    for task in ORDER[i + 1:]:
        if ledger_rows(ledger, task):
            raise SystemExit(f"refusing: {task} comes after {seg.task} and already has ledger rows")


def job_manifest(fz: Frozen, seg: Segment, job: DDJob, calset: str) -> dict[str, Any]:
    """Position -> circuit name, ``native_sha256``, qubits, and the DD state: what ``hw.submit`` is about to send."""
    return {
        "format": "dqiqm.emerald-dd/1", "task": seg.task, "device": DEVICE, "label": job.label, "dd": job.arm,
        "compilation_options": options_record(job.dd), "experiment": job.experiment, "segment_position": job.position,
        "shots": job.shots, "circuits": len(job.names), "calibration_set_id": calset, "purpose": tagged_purpose(seg, job),
        "order": [{"position": i, "circuit": n, "native_sha256": fz.placement[n]["native_sha256"],
                   "physical_qubits": fz.planned_qubits(n)} for i, n in enumerate(job.names)],
        "qubit_index_to_name": {str(k): v for k, v in fz.names_map(job).items()},
        "note": "IQM returns circuit_names [null]: this file (and meta.json's circuit_order) links c<i> in memory.npz to a circuit",
        "split": f"{SPLIT_LABEL}: not in the ML split (never finetune, validation or test)",
        "job_id": None, "raw": None, "written_utc": utc_now(),
    }


class LedgerWatch:
    """Print a new job id of ``task`` as soon as ``hw.submit`` writes its ledger row: ``scripts/batch_run.py``'s."""

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


def check_ledger_row(row: hw.LedgerRow, *, seg: Segment, job: DDJob, job_id: str, calset: str) -> dict[str, Any]:
    want = {"task": seg.task, "device": DEVICE, "calibration_set_id": calset, "qubits": ",".join(PATCH_QUBITS),
            "circuits x shots": f"{len(job.names)} x {job.shots}", "job id": job_id, "purpose": tagged_purpose(seg, job)}
    got = {"task": row.task, "device": row.device, "calibration_set_id": row.calibration_set_id, "qubits": row.qubits,
           "circuits x shots": f"{row.circuits} x {row.shots}", "job id": row.job_id, "purpose": row.purpose}
    return {k: {"want": want[k], "got": got[k], "ok": want[k] == got[k]} for k in want}


def meta_checks(meta: Mapping[str, Any], fz: Frozen, job: DDJob, calset: str) -> dict[str, bool]:
    order = [(c["position"], c["name"], c["native_sha256"]) for c in meta.get("circuit_order") or []]
    return {
        "qubit_mapping": meta["qubit_mapping"] == [{str(k): v for k, v in sorted(m.items())} for m in fz.mappings(job)],
        "physical_qubits": meta["physical_qubits"] == [fz.planned_qubits(n) for n in job.names],
        "measured_qubits": meta.get("measured_qubits") == fz.measured(job),
        "circuit_order": order == [(i, n, h) for i, (n, h) in enumerate(zip(job.names, fz.hashes(job)))],
        "compilation_options": _plain(meta.get("compilation_options")) == _plain(options_record(job.dd)),
        "calibration_set_id": meta["calibration_set_id"] == calset,
        "shots": meta["shots"] == job.shots,
        "circuits": meta["circuits"] == len(job.names),
        "success": bool(meta["success"]),
        "timeline": bool(meta.get("timeline")),
    }


def server_record(job_id: str, *, attempts: int = 3, wait_s: float = 5.0) -> dict[str, Any]:
    """The server's own record of a job, re-pulled read-only (``hw.retrieve_native``: the job's payload). Reads only."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            nj = hw.retrieve_native(job_id, DEVICE)
            p = nj.parameters
            return {"status": str(getattr(getattr(nj.data, "status", None), "value", getattr(nj.data, "status", None))),
                    "compilation_options": _plain(hw.compilation_options_record(p)), "shots": int(p.shots),
                    "calibration_set_id": str(p.calibration_set_id),
                    "native_sha256": [hw.instructions_sha256(c.instructions) for c in nj.circuits],
                    "fetched_utc": utc_now(), "via": "hw.retrieve_native(job_id, 'emerald').parameters (read-only)"}
        except Exception as e:  # noqa: BLE001 -- a read: retried, then reported
            last = e
            log(f"  server record of {job_id}: attempt {attempt + 1} raised {type(e).__name__}: {e}")
            time.sleep(wait_s)
    return {"error": f"{type(last).__name__}: {last}"}


def server_checks(rec: Mapping[str, Any], fz: Frozen, job: DDJob, calset: str) -> dict[str, bool]:
    """The server's record against the job: the DD mode and its sequences, the circuits, the shots and the calset."""
    if "error" in rec:
        return {"server record": False}
    want = _plain(options_record(job.dd))
    got = rec["compilation_options"]
    seq = lambda o: (o.get("dd_strategy") or {}).get("gate_sequences")  # noqa: E731
    return {
        "dd_mode": got.get("dd_mode") == want["dd_mode"],
        "dd_strategy sequences": seq(got) == seq(want),
        "heralding and reset": (got.get("heralding_mode"), got.get("active_reset_cycles"))
        == (want["heralding_mode"], want["active_reset_cycles"]),
        "circuits = native_sha256": rec["native_sha256"] == fz.hashes(job),
        "shots": rec["shots"] == job.shots,
        "calibration_set_id": rec["calibration_set_id"] == calset,
    }


def job_sanity(job_dir: Path, fz: Frozen, job: DDJob, refs: Mapping[str, Mapping[str, Any]],
               cal: dev.Calibration | None = None) -> list[dict[str, Any]]:
    """Every circuit of a saved job through :func:`sanity` (the pilot through :func:`pilot_checks` too)."""
    meta = json.loads((Path(job_dir) / "meta.json").read_text())
    out = []
    for i, name in enumerate(job.names):
        spec = fz.suite[name]
        try:
            raw, _ = D.load_memory(job_dir, i, circuit=spec.stim)
        except (ValueError, KeyError, IndexError) as e:
            out.append({"position": i, "circuit": name, "pass": False, "error": f"{type(e).__name__}: {e}", "checks": []})
            continue
        dets, obs = D.detection_events(spec.stim, raw)
        phys = (meta.get("physical_qubits") or [None] * len(job.names))[i]
        checks = sanity(raw, dets, refs[name], shots=job.shots, physical=phys, planned=fz.planned_qubits(name))
        if job.label == "DDP" and cal is not None and len(raw) == job.shots:
            dec, _, _ = decoder_priors(cal, spec)
            pred = D.decode(dec, dets)
            checks += pilot_checks(raw, dets, obs, refs[name], pm_failures=int((pred[:, 0] != obs[:, 0]).sum()),
                                   timeline=meta.get("timeline") or [])
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
# snapshot, plan, verify
# ---------------------------------------------------------------------------


def _resonance_check() -> Any:
    spec = importlib.util.spec_from_file_location("resonance_check", REPO / "scripts" / "resonance_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def take_snapshot() -> Path:
    """An Emerald snapshot, live and read-only: ``scripts/resonance_check.py``'s fetch and save, for Emerald alone
    (its ``main`` also writes a Garnet one, under a path this task does not own)."""
    rc = _resonance_check()
    calls: list[dict] = []
    log(f"{utc_now()} snapshot of {DEVICE} (read-only: resonance_check.fetch_snapshot)")
    devices = rc.fetch_device_list(calls)
    if DEVICE not in [d["alias"] for d in devices]:
        raise SystemExit(f"STOP: {DEVICE} is not in the device list ({[d.get('alias') for d in devices]}); {calls}")
    snap = rc.fetch_snapshot(DEVICE, calls)
    snap["device_list"] = devices
    if "architecture" not in snap:
        raise SystemExit(f"STOP: no architecture for {DEVICE}: {snap['helpers']}")
    path = rc.save_snapshot(snap)
    log(f"new snapshot: {_rel(path, REPO)}")
    return path


def cmd_snapshot(args: argparse.Namespace) -> int:
    path = take_snapshot()
    snap = load_snapshot(path)
    log(f"SNAPSHOT {_rel(path, REPO)}")
    log(f"  calibration_set_id {snap['calibration_set_id']}; server default at fetch {snap.get('default_calibration_set_id_at_fetch')}")
    log(f"  fetched {snap['fetched_utc_start']} .. {snap['fetched_utc_end']}")
    log(f"  same as the design's {EM.CALSET}: {snap['calibration_set_id'] == EM.CALSET}")
    failed = [h for h in snap.get("helpers", []) if not h.get("ok")]
    log(f"  helper calls failed: {failed or 'none'}")
    return 0


def references(fz: Frozen, jobs: Sequence[DDJob], cal: dev.Calibration) -> dict[str, dict[str, Any]]:
    return {n: reference(fz.suite[n], cal) for n in sorted({n for job in jobs for n in job.names})}


def preflight(fz: Frozen, seg: Segment, snapshot: Path | None, *, live: bool) -> dict[str, Any]:
    """Everything before the first job: the plan, the freeze, the live checks, the pinned requests, the references."""
    out: dict[str, Any] = {"task": seg.task, "device": DEVICE,
                           "jobs": [{"label": j.label, "dd": j.arm, "experiment": j.experiment, "circuits": len(j.names),
                                     "shots": j.shots, "total_shots": j.total_shots, "order": list(j.names),
                                     "compilation_options": options_record(j.dd), "purpose": tagged_purpose(seg, j)}
                                    for j in seg.jobs],
                           "allowance": {"jobs": seg.max_jobs, "shots": seg.max_shots}, "arming": seg.arming(),
                           "frozen_from": _rel(EM.DEFAULT_SNAPSHOT, REPO), "frozen_calset": fz.cal.calibration_set_id}
    out["freeze"] = {j.label: freeze_checks(fz, j) for j in seg.jobs}
    log(f"FREEZE OK: {len({n for j in seg.jobs for n in j.names})} circuits rebuilt from {out['frozen_from']} equal "
        "placement.json (native, stim and qasm hashes), native ops only, cz on calibrated couplers, clbit j = "
        "measurement j, 0 SWAPs, all on eA's 17 qubits")
    for j in seg.jobs:
        m = fz.names_map(j)
        log(f"  {j.label} (DD {j.arm}): {len(j.names)} x {j.shots}; qubit_index_to_name over {len(m)} qubits; "
            f"purpose {tagged_purpose(seg, j)!r}")
    cal = fz.cal
    if snapshot is not None:
        out["live"] = live_checks(fz, seg.jobs, snapshot)
        log(f"LIVE CALIBRATION OK in {snapshot}: {json.dumps(out['live'])}")
        cal = EM.live_calibration(snapshot)
        if live:
            calset = out["live"]["calibration_set_id"]
            out["run_requests"] = {j.label: run_request_checks(live_run_request(fz, j, calset), fz, j, calset) for j in seg.jobs}
            for label, rr in out["run_requests"].items():
                log(f"RUN REQUEST OK (read-only backend, not submitted) {label}: {rr['circuits']} circuits x {rr['shots']}, "
                    f"calset {rr['calibration_set_id']}, dd_mode {rr['options']['dd_mode']}, every circuit's instructions = "
                    f"its native_sha256, qubits eA's 17")
    out["references"] = references(fz, seg.jobs, cal)
    out["references_calibration_set_id"] = cal.calibration_set_id
    return out


def cmd_plan(args: argparse.Namespace) -> int:
    problems = check_segments(load_placement())
    for seg in SEGMENTS.values():
        log(f"{seg.task}: {seg.n_jobs} jobs, {seg.total_shots:,} shots (the brief allows {seg.max_jobs} and {seg.max_shots:,})")
        for j in seg.jobs:
            log(f"  {j.label}: DD {j.arm}, {j.experiment}, {len(j.names)} x {j.shots} = {j.total_shots:,}: {', '.join(j.names)}")
        log("  arming: " + " ".join(f"{k}={v}" for k, v in seg.arming().items()))
    log(f"total {sum(s.n_jobs for s in SEGMENTS.values())} jobs, {sum(s.total_shots for s in SEGMENTS.values()):,} shots")
    log(f"DD on: {json.dumps(options_record(True))}; ledger tag {hw.compilation_options_tag(options_record(True))!r}")
    log(f"problems: {problems or 'none'}")
    return 1 if problems else 0


def cmd_verify(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    if args.live and not args.snapshot:
        raise SystemExit("--live needs --snapshot (the calibration set to build the requests for)")
    fz = Frozen()
    out = preflight(fz, seg, args.snapshot, live=args.live)
    for name, ref in out["references"].items():
        m = ref["model"] or {}
        log(f"  ref {name}: random first {ref['random_cols']} deterministic first {ref['deterministic_cols']} "
            f"noiseless all-zero {ref['noiseless_all_zero']:.4f}; model detection {m.get('detection_fraction')}")
    if args.snapshot:
        log(f"ALLOWANCE (not armed; the check submit will run, first job): {_allowance_line(seg)}")
    return 0


def _allowance_line(seg: Segment) -> str:
    env = {**seg.arming(), "DQ_QPU_LEDGER": str(hw.ledger_path())}
    j = seg.jobs[0]
    try:
        hw.check_allowance(seg.task, len(j.names), j.shots, env=env)
    except hw.QPUGuardError as e:
        return f"REFUSED: {e}"
    return f"fits: {seg.task} usage {hw.usage(seg.task, hw.ledger_path())} + {len(j.names)} x {j.shots}"


# ---------------------------------------------------------------------------
# submit, recover
# ---------------------------------------------------------------------------


def submit_job(fz: Frozen, seg: Segment, job: DDJob, *, calset: str, ledger: Path, raw_root: Path,
               refs: Mapping[str, Mapping[str, Any]], cal: dev.Calibration, timeout: float | None) -> dict[str, Any]:
    """One ``hw.submit`` of one job. Returns what happened; a failure after the job id says how to recover."""
    rows = job_rows(ledger, seg).get(job.label, [])
    if rows:
        raise SystemExit(f"refusing: {job.label} already has ledger row(s) {[r.job_id for r in rows]}; there is no resubmission")
    man = job_manifest(fz, seg, job, calset)
    mpath = manifest_path(raw_root, seg, job, dt.datetime.now(dt.UTC).strftime("%Y%m%d"))
    if mpath.exists() and json.loads(mpath.read_text()).get("job_id"):
        raise SystemExit(f"refusing: {mpath} already names a job")
    write_json(mpath, man)
    log(f"{utc_now()} manifest {_rel(mpath, REPO)} written; submitting {job.label} (DD {job.arm}): {len(job.names)} circuits "
        f"x {job.shots} shots on {DEVICE}, calibration set {calset}")
    error, sub = None, None
    with LedgerWatch(ledger, seg.task):
        try:
            sub = hw.submit(fz.circuits(job), device=DEVICE, shots=job.shots, purpose=purpose(seg, job),
                            calibration_set_id=calset, qubit_index_to_name=fz.names_map(job),
                            circuit_compilation_options=compilation_options(job.dd), circuit_names=list(job.names),
                            native_sha256=fz.hashes(job), data_root=raw_root, timeout=timeout)
        except BaseException as e:  # noqa: BLE001 -- reported below, whatever it is
            error = e
    rows = job_rows(ledger, seg).get(job.label, [])
    job_id = rows[0].job_id if len(rows) == 1 else None
    man.update(job_id=job_id, submitted_utc=rows[0].time if rows else None)
    write_json(mpath, man)
    log(f"JOB ID {job_id}" if job_id else f"NO JOB ID: {len(rows)} ledger rows for {job.label}")
    if error is not None:
        log(f"hw.submit raised {type(error).__name__}: {error}")
        if isinstance(error, hw.JobMayExistError):
            log(f"FAIL-CLOSED ROW {error.ledger_job_id}: the job may exist. STOP and report; never retry.")
        elif job_id and not job_id.startswith(hw.UNKNOWN_JOB_PREFIX):
            log(f"RECOVER (read-only, never submit again): uv run python scripts/emerald_dd.py recover --segment {seg.task} "
                f"--job-id {job_id}")
        return {"label": job.label, "job_id": job_id, "error": f"{type(error).__name__}: {error}", "ok": False,
                "manifest": str(mpath)}
    return finish_job(fz, seg, job, Path(sub.path), calset=calset, ledger=ledger, refs=refs, cal=cal, manifest=mpath)


def finish_job(fz: Frozen, seg: Segment, job: DDJob, job_dir: Path, *, calset: str, ledger: Path,
               refs: Mapping[str, Mapping[str, Any]], cal: dev.Calibration, manifest: Path,
               server: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """After the raw data is saved: the ledger row, ``meta.json``, the server's record and every circuit's sanity."""
    meta = json.loads((job_dir / "meta.json").read_text())
    rows = [r for r in ledger_rows(ledger, seg.task) if r.job_id == meta["job_id"]]
    row_check = check_ledger_row(rows[0], seg=seg, job=job, job_id=meta["job_id"], calset=calset) if rows else {}
    mcheck = meta_checks(meta, fz, job, calset)
    server = dict(server) if server is not None else server_record(meta["job_id"])
    scheck = server_checks(server, fz, job, calset)
    results = job_sanity(job_dir, fz, job, refs, cal)
    ok_row = len(rows) == 1 and all(v["ok"] for v in row_check.values())
    ok = ok_row and all(mcheck.values()) and all(scheck.values()) and all(r["pass"] for r in results)
    man = json.loads(manifest.read_text()) if manifest.exists() else job_manifest(fz, seg, job, calset)
    man.update(job_id=meta["job_id"], raw=_rel(job_dir, RECORD), ledger_rows=len(rows), ledger_check=row_check,
               meta_check=mcheck, server=server, server_check=scheck, sanity=results, ok=ok, checked_utc=utc_now())
    write_json(manifest, man)
    log(f"raw: {_rel(job_dir, REPO)}")
    log(f"ledger row ({len(rows)}): " + ("OK" if ok_row else json.dumps(row_check)))
    log("meta.json: " + ("OK" if all(mcheck.values()) else json.dumps(mcheck)))
    log(f"server record: dd_mode {server.get('compilation_options', {}).get('dd_mode')}, "
        + ("OK" if all(scheck.values()) else json.dumps({"check": scheck, "server": server})))
    for r in results:
        if not r["pass"]:
            log(f"  SANITY FAIL position {r['position']} {r['circuit']}: "
                + json.dumps([c for c in r["checks"] if not c["pass"]] or r.get("error")))
        if job.label == "DDP":
            for c in r["checks"]:
                log(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}: {json.dumps(c['value'])}  band {c['band']}"
                    + (f"  {json.dumps(c['info'])}" if c.get("info") else ""))
    log(f"sanity: {sanity_line(results)}")
    return {"label": job.label, "job_id": meta["job_id"], "path": str(job_dir), "ledger_rows": len(rows), "ok": ok,
            "manifest": str(manifest), "sanity": sanity_line(results)}


def resume_state(fz: Frozen, seg: Segment, job: DDJob, *, ledger: Path, raw_root: Path,
                 refs: Mapping[str, Mapping[str, Any]], cal: dev.Calibration) -> str:
    """A job's state before ``submit`` reaches it: "new", "done" (in the ledger, saved, checked OK), "failed" or
    "raw missing". A job saved but never checked (the process died in between) is checked now."""
    done = job_rows(ledger, seg).get(job.label, [])
    if not done:
        return "new"
    if len(done) > 1 or done[0].job_id.startswith(hw.UNKNOWN_JOB_PREFIX):
        log(f"{job.label}: ledger rows {[r.job_id for r in done]}: report it")
        return "failed"
    jd = find_job_dir(raw_root, seg.task, done[0].job_id)
    log(f"{job.label}: already submitted as {done[0].job_id}" + ("" if jd else " -- RAW MISSING: run recover"))
    if jd is None:
        return "raw missing"
    mpath = manifest_path(raw_root, seg, job, done[0].time[:10].replace("-", ""))
    checked = json.loads(mpath.read_text()).get("ok") if mpath.exists() else None
    if checked is None:
        checked = finish_job(fz, seg, job, jd, calset=done[0].calibration_set_id, ledger=ledger, refs=refs, cal=cal,
                             manifest=mpath)["ok"]
    return "done" if checked else "failed"


def cmd_submit(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    check_arming(seg, os.environ)
    ledger = hw.ledger_path()
    check_order(seg, ledger, RAW_ROOT)
    snapshot = Path(args.snapshot)
    calset = load_snapshot(snapshot)["calibration_set_id"]
    fz = Frozen()
    pre = preflight(fz, seg, snapshot, live=False)
    refs = pre["references"]
    cal = EM.live_calibration(snapshot)
    day = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    pre_path = RAW_ROOT / f"{day}-{seg.task}" / "preflight.json"
    if not pre_path.exists():
        write_json(pre_path, pre)
    default_now = hw.calibration_set_id(DEVICE)
    if default_now != calset:
        raise SystemExit(f"refusing: the server's default calibration set is now {default_now}, the snapshot's {calset}; "
                         "take a new snapshot")
    calsets = [{"calibration_set_id": calset, "snapshot": _rel(snapshot, REPO), "from_job": seg.jobs[0].label}]
    for job in seg.jobs:
        state = resume_state(fz, seg, job, ledger=ledger, raw_root=RAW_ROOT, refs=refs, cal=cal)
        if state == "raw missing":
            return 5
        if state == "failed":
            log(f"STOP: {job.label} failed its checks; the rest of {seg.task} is not submitted. Report it.")
            return 4
        if state == "done":
            continue
        now = hw.calibration_set_id(DEVICE)
        if now != calset:
            log(f"{utc_now()} the server's default calibration set moved: {calset} -> {now}; a new snapshot and the live checks")
            new = take_snapshot()
            try:
                live = live_checks(fz, seg.jobs, new)
            except AssertionError as e:
                log(str(e))
                return 6
            if live["calibration_set_id"] != now:
                raise SystemExit(f"STOP: the new snapshot is {live['calibration_set_id']}, the server default {now}")
            calset = now
            cal = EM.live_calibration(new)
            refs = references(fz, seg.jobs, cal)
            calsets.append({"calibration_set_id": calset, "snapshot": _rel(new, REPO), "from_job": job.label})
        try:
            rr = run_request_checks(live_run_request(fz, job, calset), fz, job, calset)
        except AssertionError as e:
            log(f"STOP: the run request for {job.label} failed its check: {e}")
            return 7
        log(f"RUN REQUEST OK (read-only backend, not submitted) {job.label}: {json.dumps(rr)}")
        res = submit_job(fz, seg, job, calset=calset, ledger=ledger, raw_root=RAW_ROOT, refs=refs, cal=cal,
                         timeout=args.timeout)
        log(f"{job.label} " + ("OK" if res["ok"] else "NOT OK") + ": " + json.dumps(res))
        if not res["ok"]:
            log(f"STOP: {job.label} failed a check; the rest of {seg.task} is not submitted. Report it.")
            return 4
    rows = job_rows(ledger, seg)
    seg_file = RAW_ROOT / f"{day}-{seg.task}" / "segment.json"
    write_json(seg_file, {
        "format": "dqiqm.emerald-dd/1", "task": seg.task, "device": DEVICE,
        "allowance": {"jobs": seg.max_jobs, "shots": seg.max_shots}, "armed": seg.arming(), "calibration_sets": calsets,
        "before": _rel(snapshot, REPO),
        "jobs": [{"label": j.label, "dd": j.arm, "position": j.position, "job_id": rows[j.label][0].job_id,
                  "ledger_time": rows[j.label][0].time, "calibration_set_id": rows[j.label][0].calibration_set_id,
                  "raw": _rel(find_job_dir(RAW_ROOT, seg.task, rows[j.label][0].job_id), RECORD),
                  "order": [{"position": i, "circuit": n, "native_sha256": fz.placement[n]["native_sha256"]}
                            for i, n in enumerate(j.names)]} for j in seg.jobs],
        "written_utc": utc_now()})
    n = len(ledger_rows(ledger, seg.task))
    shots = sum(r.total_shots for r in ledger_rows(ledger, seg.task))
    log(f"SEGMENT SUBMITTED {seg.task}: {n} jobs, {shots:,} shots in the ledger (armed {seg.n_jobs}, {seg.total_shots:,}); "
        f"{_rel(seg_file, REPO)}")
    return 0 if (n, shots) == (seg.n_jobs, seg.total_shots) else 8


def recover_job(fz: Frozen, seg: Segment, job_id: str, *, ledger: Path, raw_root: Path,
                timeout: float | None) -> tuple[DDJob, Path]:
    """Re-pull a job read-only and save it in ``save_raw``'s layout, with what was sent. Never submits."""
    rows = [r for r in ledger_rows(ledger, seg.task) if r.job_id == job_id]
    if len(rows) != 1:
        raise SystemExit(f"refusing: job {job_id} has {len(rows)} {seg.task} rows in {ledger}")
    label = label_of(rows[0])
    job = next((j for j in seg.jobs if j.label == label), None)
    if job is None:
        raise SystemExit(f"refusing: the ledger row's purpose names {label!r}, not a job of {seg.task}")
    if rows[0].purpose != tagged_purpose(seg, job):
        raise SystemExit(f"refusing: the ledger row's purpose {rows[0].purpose!r} is not {job.label}'s {tagged_purpose(seg, job)!r}")
    day = rows[0].time[:10].replace("-", "")
    target = Path(raw_root) / f"{day}-{seg.task}" / job_id
    if target.exists():
        raise SystemExit(f"refusing: {target} exists ({sorted(x.name for x in target.iterdir())}); it is never overwritten")
    result = hw.retrieve(job_id, DEVICE, timeout=timeout)
    order = [{"position": i, "name": n, "native_sha256": h} for i, (n, h) in enumerate(zip(job.names, fz.hashes(job)))]
    path = hw.save_raw(result, task=seg.task, device=DEVICE, qubit_mapping=fz.mappings(job), shots=job.shots,
                       physical_qubits=[fz.planned_qubits(n) for n in job.names], circuit_order=order,
                       compilation_options=options_record(job.dd), purpose=rows[0].purpose, day=day, data_root=raw_root)
    return job, path


def cmd_recover(args: argparse.Namespace) -> int:
    seg = SEGMENTS[args.segment]
    fz = Frozen()
    ledger = hw.ledger_path()
    job, path = recover_job(fz, seg, args.job_id, ledger=ledger, raw_root=RAW_ROOT, timeout=args.timeout)
    log(f"RECOVERED (hw.retrieve + hw.save_raw, read-only): {path}")
    row = next(r for r in ledger_rows(ledger, seg.task) if r.job_id == args.job_id)
    cal = EM.live_calibration(Path(args.snapshot)) if args.snapshot else fz.cal
    refs = references(fz, [job], cal)
    res = finish_job(fz, seg, job, path, calset=row.calibration_set_id, ledger=ledger, refs=refs, cal=cal,
                     manifest=manifest_path(RAW_ROOT, seg, job, row.time[:10].replace("-", "")))
    log(f"{job.label} " + ("OK" if res["ok"] else "NOT OK") + ": " + json.dumps(res))
    return 0 if res["ok"] else 4


# ---------------------------------------------------------------------------
# The A/B statistics: pure functions of counts.
# ---------------------------------------------------------------------------


def proportion(k: int, n: int) -> dict[str, Any]:
    lo, hi = stats.wilson(k, n)
    return {"k": int(k), "n": int(n), "p": k / n, "wilson95": [float(lo), float(hi)]}


def newcombe(k1: int, n1: int, k2: int, n2: int) -> dict[str, Any]:
    """p1 - p2 with Newcombe's hybrid score 95% interval (from the two Wilson intervals), and the pooled z."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = stats.wilson(k1, n1)
    l2, u2 = stats.wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return {"diff": d, "ci95": [lo, hi], "z": d / se if se > 0 else float("nan")}


def eps_fit(points: Sequence[tuple[int, int, int]]) -> dict[str, Any]:
    """eps_L by maximum likelihood (A = 1 and A free, ``stats.fit_eps``) from (rounds, failures, shots) points."""
    r = np.array([p[0] for p in points])
    k = np.array([p[1] for p in points])
    n = np.array([p[2] for p in points])
    out = {}
    for name, free in (("fixed_A", False), ("free_A", True)):
        if len(np.unique(r)) < (2 if free else 1):
            continue
        f = stats.fit_eps(r, k, n, free_amplitude=free, bootstrap=EPS_BOOTSTRAP, seed=SEED)
        out[name] = {"eps": f.eps, "eps_ci_bootstrap": list(f.eps_ci_bootstrap), "amplitude": f.amplitude,
                     "chi2_per_dof": f.chi2_per_dof}
    return out


def eps_difference(off: Sequence[tuple[int, int, int]], on: Sequence[tuple[int, int, int]],
                   *, reps: int = EPS_BOOTSTRAP, seed: int = SEED) -> dict[str, Any]:
    """eps(off) - eps(on) (A = 1) and its 95% interval by a seeded parametric bootstrap of both arms' counts."""
    def fit(pts: Sequence[tuple[int, int, int]], k: np.ndarray | None = None) -> float:
        r = np.array([p[0] for p in pts])
        n = np.array([p[2] for p in pts])
        k = np.array([p[1] for p in pts]) if k is None else k
        return stats.fit_eps(r, k, n, free_amplitude=False, bootstrap=0).eps

    d = fit(off) - fit(on)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        ks = [rng.binomial([p[2] for p in arm], [p[1] / p[2] for p in arm]) for arm in (off, on)]
        draws.append(fit(off, ks[0]) - fit(on, ks[1]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": d, "ci95_bootstrap": [float(lo), float(hi)], "reps": reps, "seed": seed,
            "rule": "eps(off) - eps(on), A = 1; each arm's failures redrawn as Binomial(n, k/n) and refitted"}


def pair_table(per_job: Mapping[str, tuple[int, int]], on: Sequence[str], off: Sequence[str]) -> dict[str, Any]:
    """One (basis, R) cell of the A/B: the pooled arms, their difference, and the within-arm drift."""
    k_on, n_on = sum(per_job[j][0] for j in on), sum(per_job[j][1] for j in on)
    k_off, n_off = sum(per_job[j][0] for j in off), sum(per_job[j][1] for j in off)
    out = {"on": proportion(k_on, n_on), "off": proportion(k_off, n_off),
           "off_minus_on": newcombe(k_off, n_off, k_on, n_on),
           "jobs": {j: proportion(*per_job[j]) for j in (*on, *off)}}
    if len(on) == 2:
        out["drift_on"] = {"pair": f"{on[0]} - {on[1]}", **newcombe(*per_job[on[0]], *per_job[on[1]])}
    if len(off) == 2:
        out["drift_off"] = {"pair": f"{off[0]} - {off[1]}", **newcombe(*per_job[off[0]], *per_job[off[1]])}
    return out


# ---------------------------------------------------------------------------
# analyze: the records and the A/B
# ---------------------------------------------------------------------------


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def timeline_numbers(timeline: Sequence[Mapping[str, Any]], *, shots: int) -> dict[str, Any]:
    """Spans of the server's status timeline (``scripts/batch_run.py``'s)."""
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


def per_round(dets: np.ndarray, t: np.ndarray) -> dict[str, float]:
    return {str(k): float(dets[:, t == k].mean()) for k in sorted(set(t.tolist()))}


def run_id(job: DDJob, name: str) -> str:
    return f"dd_{job.label}_{job.arm}_{name}"


def analyze(snapshots: Sequence[Path], *, root: Path, ledger: Path, fz: Frozen | None = None) -> dict[str, Any]:
    """Write every run's dataset, derived features and checks, then the LER records and the A/B."""
    fz = fz or Frozen()
    cals: dict[str, tuple[Path, dev.Calibration]] = {}
    for p in snapshots:
        c = EM.live_calibration(Path(p))
        cals.setdefault(c.calibration_set_id, (Path(p), c))
    snap_rel = {"before": _rel(Path(snapshots[0]), root), "after": _rel(Path(snapshots[-1]), root)}
    runs: list[dict[str, Any]] = []
    jobs_out: list[dict[str, Any]] = []
    for seg in SEGMENTS.values():
        rows = job_rows(ledger, seg)
        for job in seg.jobs:
            jr = rows.get(job.label, [])
            if not jr:
                log(f"{seg.task} {job.label}: not in the ledger; skipped")
                continue
            if len(jr) != 1:
                raise SystemExit(f"{job.label}: {len(jr)} ledger rows")
            job_dir = find_job_dir(RAW_ROOT, seg.task, jr[0].job_id)
            if job_dir is None:
                raise SystemExit(f"{job.label}: job {jr[0].job_id} has no raw data; recover it first")
            meta = json.loads((job_dir / "meta.json").read_text())
            if _plain(meta.get("compilation_options")) != _plain(options_record(job.dd)):
                raise SystemExit(f"{job.label}: meta.json's compilation_options are not DD {job.arm}'s")
            if meta["calibration_set_id"] not in cals:
                raise SystemExit(f"{job.label} ran on {meta['calibration_set_id']}; pass its snapshot (have {sorted(cals)})")
            snap_path, cal = cals[meta["calibration_set_id"]]
            durations = dev.Durations.from_calibration(cal)
            mpath = manifest_path(RAW_ROOT, seg, job, jr[0].time[:10].replace("-", ""))
            man = json.loads(mpath.read_text()) if mpath.exists() else {}
            tl = timeline_numbers(meta.get("timeline") or [], shots=job.total_shots)
            jobs_out.append({"task": seg.task, "label": job.label, "dd": job.arm, "job_id": meta["job_id"],
                             "calibration_set_id": meta["calibration_set_id"], "circuits": len(job.names), "shots": job.shots,
                             "total_shots": job.total_shots, "ledger_time": jr[0].time, "ledger_purpose": jr[0].purpose,
                             "timeline": tl, "raw": _rel(job_dir, root), "checked_ok": man.get("ok"),
                             "server_dd_mode": (man.get("server") or {}).get("compilation_options", {}).get("dd_mode")})
            log(f"{job.label} {meta['job_id']} (DD {job.arm}): {len(job.names)} circuits, timeline {json.dumps(tl)}")
            for i, name in enumerate(job.names):
                spec = fz.suite[name]
                entry = fz.placement[name]
                raw, _ = D.load_memory(job_dir, i, circuit=spec.stim)
                rid = run_id(job, name)
                dec, prior, note = decoder_priors(cal, spec)
                dec_e, prior_e, note_e = decoder_priors(cal, spec, t2="echo")
                files = {"stim": f"circuits/{spec.experiment}/{name}.stim", "qasm": f"circuits/{spec.experiment}/{name}.qasm",
                         "qasm_sha256": entry["qasm_sha256"], "native_sha256": entry["native_sha256"]}
                labels = [SPLIT_LABEL] * len(raw)
                split_info = {"rule": "the DD A/B (emerald-dd): not in the ML split; no shot is finetune, validation or test",
                              "job": job.label, "role": SPLIT_LABEL, "counts": {SPLIT_LABEL: len(raw)}}
                raw_ref = {"path": _rel(job_dir / "memory.npz", root), "key": f"c{i}", "meta": _rel(job_dir / "meta.json", root),
                           "task": seg.task, "job": job.label, "dd": job.arm,
                           "compilation_options": meta["compilation_options"],
                           "manifest": _rel(mpath, root) if mpath.exists() else None}
                ds, numbers = IR.write_dataset(
                    root, spec, raw, run_id=rid, batch=0, job_id=meta["job_id"], position_in_job=i, arch=fz.suite.arch,
                    cal=cal, durations=durations, circuit_files=files, raw_ref=raw_ref,
                    calibration_paths={**snap_rel, "job": _rel(snap_path, root)}, decoder_circuit=dec,
                    decoder_note=f"{note}. DD {job.arm} ({job.label}; compilation options in raw.compilation_options)",
                    split_labels=labels, split_info=split_info, simulated=False,
                    extra={"device": DEVICE, "job_timeline": meta.get("timeline")},
                )
                der, _ = IR.write_derived(root, spec, rid, ds)
                tag_json(der / "summary.json", label="HARDWARE", device=DEVICE, task=seg.task, job=job.label, dd=job.arm,
                         priors=PRIORS["ramsey"])
                problems = IR.check_run(ds, der)
                dets = IR.load_array(ds, "dets")
                obs = IR.load_array(ds, "obs")
                with open(ds / "detectors.csv") as f:
                    t = np.array([int(r["round"]) for r in csv.DictReader(f)])
                pred_e = D.decode(dec_e, dets)
                k_e = int((pred_e[:, 0] != obs[:, 0]).sum())
                ref = reference(spec, cal)
                checks = sanity(raw, dets, ref, shots=job.shots, physical=(meta.get("physical_qubits") or [None] * 99)[i],
                                planned=fz.planned_qubits(name))
                body = {"format": IR.FORMAT, "label": "HARDWARE", "task": seg.task, "device": DEVICE, "job": job.label,
                        "dd": job.arm, "compilation_options": meta["compilation_options"], "job_id": meta["job_id"],
                        "position_in_job": i, "circuit": name, "basis": spec.params["basis"], "rounds": spec.rounds,
                        "delay_ns": spec.params.get("delay_ns", 0.0), "calibration_set_id": meta["calibration_set_id"],
                        "decoders": {
                            "ramsey": {"priors": PRIORS["ramsey"], "note": note, "prior": prior,
                                       "pm_failures": numbers["pm_failures"], "shots": numbers["shots"],
                                       "p_l": proportion(numbers["pm_failures"], numbers["shots"])},
                            "echo": {"priors": PRIORS["echo"], "note": note_e, "prior": prior_e, "pm_failures": k_e,
                                     "shots": numbers["shots"], "p_l": proportion(k_e, numbers["shots"])}},
                        "obs_flips": proportion(numbers["obs_flips"], numbers["shots"]),
                        "detection_fraction": numbers["det_rate"], "per_round_detection_rate": per_round(dets, t),
                        "sanity": checks, "sanity_pass": all(c["pass"] for c in checks), "model_with_idle": ref["model"],
                        "numbers": numbers, "record_problems": problems, "created_utc": utc_now(), "commit": hw.git_head()}
                write_json(der / "dd_checks.json", body)
                runs.append({"seg": seg, "job": job, "spec": spec, "run_id": rid, "dataset": ds, "body": body,
                             "dets": dets, "t": t, "note": note})
                if problems or not body["sanity_pass"]:
                    log(f"  {rid}: problems {problems}; sanity {body['sanity_pass']}")
    ler = write_ler_records(runs, root)
    ab = {"E1": ab_summary(runs, "E1", ler), "E4": ab_summary(runs, "E4", ler)}
    for exp, body in ab.items():
        if body is None:
            continue
        out_dir = root / "ler" / exp / "dd_ab"
        write_json(out_dir / "ab.json", {"format": IR.FORMAT, "label": "HARDWARE", "device": DEVICE, "experiment": exp,
                                         "jobs": [j for j in jobs_out if (exp == "E1") == (j["label"] not in ("E4A", "E4B"))],
                                         **body, "snapshots": [_rel(Path(p), REPO) for p in snapshots],
                                         "created_utc": utc_now(), "commit": hw.git_head()})
        log(f"wrote {_rel(out_dir / 'ab.json', REPO)}")
    return {"jobs": jobs_out, "runs": len(runs),
            "record_problems": {r["run_id"]: r["body"]["record_problems"] for r in runs if r["body"]["record_problems"]},
            "sanity_failures": [r["run_id"] for r in runs if not r["body"]["sanity_pass"]], "ler": ler}


def _points(rs: Sequence[Mapping[str, Any]], root: Path, decoder: str = "ramsey") -> list[dict[str, Any]]:
    return [{"circuit": r["spec"].name, "dataset": _rel(r["dataset"], root), "rounds": r["spec"].rounds,
             "shots": r["body"]["numbers"]["shots"], "failures": r["body"]["decoders"][decoder]["pm_failures"],
             "raw_failures": r["body"]["numbers"]["obs_flips"]} for r in rs]


def write_ler_records(runs: Sequence[Mapping[str, Any]], root: Path) -> dict[str, Any]:
    """LER records: per job and basis, and per arm pooled (E1: A1+A2, B1+B2; E4: E4A, E4B); the pilot without a fit."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in runs:
        spec, job = r["spec"], r["job"]
        b = spec.params["basis"]
        if spec.experiment == "E1":
            groups[("E1", f"dd_{b}_eA_{job.label}_{job.arm}")].append(r)
            if job.label != "DDP":
                groups[("E1", f"dd_{b}_eA_{job.arm}")].append(r)
        else:
            groups[("E4", f"dd_{b}_eA_t5000ns_{job.arm}")].append(r)
    out = {}
    for (exp, rid), rs in sorted(groups.items()):
        points = _points(rs, root)
        fit = len({p["rounds"] for p in points}) >= 2
        job = rs[0]["job"]
        title = (f"HARDWARE (emerald): {exp} d=3 {rs[0]['spec'].params['basis']} basis, patch eA, DD {job.arm}"
                 + (f", {', '.join(sorted({r['job'].label for r in rs}))}" if exp == "E1" else ", 5 us idle between rounds"))
        path, res = IR.write_ler(root, exp, rid, points, title=title, label=rid, fit=fit)
        tag_json(path / "ler_fits.json", label="HARDWARE", device=DEVICE, dd=job.arm,
                 jobs=sorted({r["job"].label for r in rs}), priors=PRIORS["ramsey"],
                 decoder="pymatching on the noisy copy of the circuit that ran", noise=rs[0]["note"],
                 zero_failure_points=[x["rounds"] for x in points if x["failures"] == 0])
        out[rid] = {"experiment": exp, "points": len(points), "shots": sum(x["shots"] for x in points),
                    **({"eps_fixed_A": res["fixed_A"]["eps"], "eps_fixed_A_ci": res["fixed_A"]["eps_ci_bootstrap"],
                        "eps_free_A": res["free_A"]["eps"], "amplitude": res["free_A"]["amplitude"]} if "free_A" in res else {})}
        if "free_A" in res:
            log(f"LER {exp} {rid}: eps(A=1) {res['fixed_A']['eps']:.4f} {res['fixed_A']['eps_ci_bootstrap']}, "
                f"eps(free A) {res['free_A']['eps']:.4f}, A {res['free_A']['amplitude']:.3f}")
    return out


def ab_summary(runs: Sequence[Mapping[str, Any]], experiment: str, ler: Mapping[str, Any]) -> dict[str, Any] | None:
    """The paired A/B of one experiment: per (basis, R) cells for both decoders and the undecoded flips, the
    detection fractions per round, and eps per arm."""
    rs = [r for r in runs if r["spec"].experiment == experiment and r["job"].label != "DDP"]
    if not rs:
        return None
    on = sorted({r["job"].label for r in rs if r["job"].dd}, key=lambda x: (len(x), x))
    off = sorted({r["job"].label for r in rs if not r["job"].dd}, key=lambda x: (len(x), x))
    if not on or not off:
        return {"incomplete": True, "on": on, "off": off}
    cells: dict[str, dict[str, Any]] = defaultdict(dict)
    for basis in ("Z", "X"):
        for rounds in sorted({r["spec"].rounds for r in rs}):
            sel = {r["job"].label: r for r in rs if r["spec"].params["basis"] == basis and r["spec"].rounds == rounds}
            if set(sel) != set(on) | set(off):
                continue
            cell: dict[str, Any] = {"circuit": next(iter(sel.values()))["spec"].name}
            for dec in ("ramsey", "echo"):
                cell[dec] = pair_table({j: (b["body"]["decoders"][dec]["pm_failures"], b["body"]["numbers"]["shots"])
                                        for j, b in sel.items()}, on, off)
            cell["obs_flips"] = pair_table({j: (b["body"]["numbers"]["obs_flips"], b["body"]["numbers"]["shots"])
                                            for j, b in sel.items()}, on, off)
            dets = {arm: np.vstack([sel[j]["dets"] for j in js]) for arm, js in (("on", on), ("off", off))}
            t = sel[on[0]]["t"]
            cell["detection"] = {
                "on": float(dets["on"].mean()), "off": float(dets["off"].mean()),
                "off_minus_on": float(dets["off"].mean() - dets["on"].mean()),
                "per_job": {j: float(sel[j]["dets"].mean()) for j in (*on, *off)},
                "per_round": {arm: per_round(d, t) for arm, d in dets.items()},
            }
            cells[basis][str(rounds)] = cell
    eps: dict[str, Any] = {}
    for basis in ("Z", "X"):
        arms = {}
        for arm, js in (("on", on), ("off", off)):
            for dec in ("ramsey", "echo"):
                pts = [(r["spec"].rounds, r["body"]["decoders"][dec]["pm_failures"], r["body"]["numbers"]["shots"])
                       for r in rs if r["spec"].params["basis"] == basis and r["job"].label in js]
                arms[f"{arm}_{dec}"] = pts
        eps[basis] = {k: eps_fit(v) for k, v in arms.items()}
        for dec in ("ramsey", "echo"):
            eps[basis][f"off_minus_on_{dec}"] = eps_difference(arms[f"off_{dec}"], arms[f"on_{dec}"])
        if len(on) == 2 and len(off) == 2:
            per_job = {j: [(r["spec"].rounds, r["body"]["decoders"]["ramsey"]["pm_failures"], r["body"]["numbers"]["shots"])
                           for r in rs if r["spec"].params["basis"] == basis and r["job"].label == j] for j in (*on, *off)}
            eps[basis]["per_job_ramsey"] = {j: eps_fit(p).get("fixed_A") for j, p in per_job.items()}
            eps[basis]["drift_on_ramsey"] = eps_difference(per_job[on[0]], per_job[on[1]])
            eps[basis]["drift_off_ramsey"] = eps_difference(per_job[off[0]], per_job[off[1]])
    return {"on_jobs": on, "off_jobs": off, "cells": dict(cells), "eps": eps,
            "ler_records": {k: v for k, v in ler.items() if v["experiment"] == experiment},
            "decoders": {"ramsey": PRIORS["ramsey"] + " on both arms: the batch's decoder; the headline",
                         "echo": PRIORS["echo"] + " on both arms: the variant the design names for DD shots; reported, "
                                 "not chosen"},
            "stat_rules": {"p": "Wilson 95%", "diff": "Newcombe hybrid score 95% (from the two Wilson intervals); z pooled",
                           "drift": "the same, between the two jobs of one arm (ABBA: A1 - A2, B1 - B2)",
                           "eps": "stats.fit_eps maximum likelihood, bootstrap 95% (1000, seed 2026)"}}


#: The batches' committed records (branch batches), read with ``git show``: the DD-off reference per stabilizer.
BATCH_COMMIT = "47eb36a"
BATCH_RUNS = {"E1": (1, 2, 3), "E4": (2,)}


def stabilizer_rates(dets: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, str]], on: Sequence[str],
                     off: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per stabilizer: its detection rate per arm (pooled over the arm's jobs), per round and per job, and off - on."""
    out: dict[str, dict[str, Any]] = {}
    for s in dict.fromkeys(r["stabilizer"] for r in rows):
        idx = [i for i, r in enumerate(rows) if r["stabilizer"] == s]
        r0 = rows[idx[0]]
        e: dict[str, Any] = {"type": r0["type"], "ancilla": r0["ancilla_qb"], "data": r0["data_qbs"].split(";"),
                             "rounds": [int(rows[i]["round"]) for i in idx]}
        for arm, js in (("on", on), ("off", off)):
            d = np.vstack([dets[j][:, idx] for j in js])
            e[arm] = {"rate": float(d.mean()), "events": int(d.sum()), "trials": int(d.size),
                      "per_round": [float(x) for x in d.mean(axis=0)]}
        e["off_minus_on"] = e["off"]["rate"] - e["on"]["rate"]
        e["per_job"] = {j: float(dets[j][:, idx].mean()) for j in (*on, *off)}
        out[s] = e
    return out


def batch_stabilizer_rates(exp: str, name: str) -> dict[str, Any] | None:
    """The batch's DD-off per-stabilizer rates for the same circuit (``git show {BATCH_COMMIT}:...``), or None."""
    import subprocess

    runs = {}
    for b in BATCH_RUNS[exp]:
        base = f"InfoRecord/emerald/derived/{exp}/b{b}_{name}"
        try:
            text = subprocess.run(["git", "show", f"{BATCH_COMMIT}:{base}/rate_by_round.csv"], cwd=REPO,
                                  capture_output=True, text=True, check=True).stdout
            meta = json.loads(subprocess.run(["git", "show", f"{BATCH_COMMIT}:{base}/batch_checks.json"], cwd=REPO,
                                             capture_output=True, text=True, check=True).stdout)
        except (OSError, subprocess.CalledProcessError):
            return None
        runs[f"b{b}"] = {"job_id": meta["job_id"], "rows": list(csv.DictReader(text.splitlines()))}
    stabs = sorted({r["stabilizer"] for v in runs.values() for r in v["rows"]})
    return {"source": f"git show {BATCH_COMMIT}:InfoRecord/emerald/derived/{exp}/b<k>_{name}/rate_by_round.csv",
            "jobs": {k: v["job_id"] for k, v in runs.items()},
            "rate": {s: float(np.mean([float(r["rate"]) for v in runs.values() for r in v["rows"] if r["stabilizer"] == s]))
                     for s in stabs}}


def first_readout_misread(raw: Mapping[str, np.ndarray], spec: Any, on: Sequence[str], off: Sequence[str]) -> dict[str, Any]:
    """Per ancilla with a deterministic first readout: how often it misreads, per arm and per job (decoder-free)."""
    ref = reference(spec, None, model=False)
    meas_q = C.measured_qubits(spec.stim)
    out = {}
    for col, want in zip(ref["deterministic_cols"], ref["deterministic_values"]):
        q = spec.physical[meas_q[col]]
        e: dict[str, Any] = {"column": col, "expected": want}
        for arm, js in (("on", on), ("off", off)):
            x = np.concatenate([raw[j][:, col] for j in js])
            e[arm] = proportion(int((x != want).sum()), len(x))
        e["off_minus_on"] = newcombe(e["off"]["k"], e["off"]["n"], e["on"]["k"], e["on"]["n"])
        e["per_job"] = {j: float((raw[j][:, col] != want).mean()) for j in (*on, *off)}
        out[q] = e
    return out


def stabilizers(fz: Frozen, root: Path = RECORD) -> dict[str, Any]:
    """Decoder-free, from the written datasets: per circuit and stabilizer, the detection rate on against off (with the
    data qubits' frame through the readout window, ``device.readout_frames``), the batch's DD-off rate of the same
    circuit, and each deterministic first readout's misread rate."""
    main = SEGMENTS["emerald-dd"]
    out: dict[str, Any] = {}
    for exp, circuits in (("E1", E1_CIRCUITS), ("E4", E4_CIRCUITS)):
        jobs = [j for j in main.jobs if j.experiment == exp]
        on, off = [j.label for j in jobs if j.dd], [j.label for j in jobs if not j.dd]
        body: dict[str, Any] = {}
        for name in circuits:
            spec = fz.suite[name]
            frames = {spec.physical[q]: "".join(f[p] for p in "XYZ") for q, f in dev.readout_frames(spec).items()}
            dets, raw, rows = {}, {}, None
            for job in jobs:
                ds = root / "datasets" / exp / run_id(job, name)
                dets[job.label], raw[job.label] = IR.load_array(ds, "dets"), IR.load_array(ds, "raw_meas")
                with open(ds / "detectors.csv") as f:
                    rows = list(csv.DictReader(f))
            st = stabilizer_rates(dets, rows, on, off)
            batch = batch_stabilizer_rates(exp, name)
            # the priors' own model (SIMULATED, the committed calibration 23e12f6b): what Ramsey or echo T2 alone gives
            model = {}
            for t2 in ("ramsey", "echo"):
                noisy = decoder_priors(fz.cal, spec, t2)[0]
                d, _ = D.detection_events(spec.stim, noisy.compile_sampler(seed=SEED).sample(MODEL_SHOTS).astype(np.uint8))
                model[t2] = stabilizer_rates({"m": d}, rows, ["m"], ["m"])
            for s, e in st.items():
                e["data_frames"] = {q: frames.get(q) for q in e["data"]}
                e["batch_dd_off"] = None if batch is None else batch["rate"].get(s)
                e["model_ramsey"] = model["ramsey"][s]["on"]["rate"]
                e["model_echo"] = model["echo"][s]["on"]["rate"]
            body[name] = {"basis": spec.params["basis"], "rounds": spec.rounds, "stabilizers": st,
                          "batch_jobs": None if batch is None else batch["jobs"],
                          "batch_source": None if batch is None else batch["source"],
                          "first_readout_misread": first_readout_misread(raw, spec, on, off)}
            big = {s: round(e["off_minus_on"], 4) for s, e in st.items() if abs(e["off_minus_on"]) >= 0.02}
            log(f"{name}: off - on per stabilizer, |d| >= 0.02: {big}; first-readout misread off - on: "
                + json.dumps({q: round(v["off_minus_on"]["diff"], 4) for q, v in body[name]["first_readout_misread"].items()}))
        out[exp] = {"on_jobs": on, "off_jobs": off, "circuits": body,
                    "frame_note": "data_frames: the physical Pauli -> Stim Pauli map of each data qubit through the readout "
                                  "window (device.readout_frames): 'XYZ' the identity, 'ZYX' a Hadamard frame, where a "
                                  "physical Z (dephasing) acts as Stim's X and so flips Z-type detectors and the Z logical"}
    return out


def cmd_stabilizers(args: argparse.Namespace) -> int:
    fz = Frozen()
    res = stabilizers(fz, Path(args.root))
    for exp, body in res.items():
        write_json(Path(args.root) / "ler" / exp / "dd_ab" / "by_stabilizer.json",
                   {"format": IR.FORMAT, "label": "HARDWARE", "device": DEVICE, "experiment": exp, **body,
                    "rule": "decoder-free: a stabilizer's rate is the mean of its detectors' events over shots and rounds; "
                            "on pools the DD-on jobs, off the DD-off jobs; a misread is a deterministic first readout "
                            "(round 1's ancilla readout of the basis's own stabilizer type) that differs from the noiseless value",
                    "created_utc": utc_now(), "commit": hw.git_head()})
        log(f"wrote {_rel(Path(args.root) / 'ler' / exp / 'dd_ab' / 'by_stabilizer.json', REPO)}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    res = analyze([Path(p) for p in args.snapshots], root=Path(args.root), ledger=Path(args.ledger))
    log(f"ANALYZED: {res['runs']} runs; record problems {res['record_problems'] or 'none'}; "
        f"sanity failures {res['sanity_failures'] or 'none'}; {len(res['ler'])} LER records")
    return 0 if not res["record_problems"] and not res["sanity_failures"] else 1


# ---------------------------------------------------------------------------
# predict: the circuits' structure, and what Stim predicts for DD off (Ramsey T2) and on (echo T2)
# ---------------------------------------------------------------------------


def predict(fz: Frozen, *, shots: int = MODEL_SHOTS, seed: int = SEED) -> dict[str, Any]:
    """SIMULATED: per circuit, the noiseless and DEM checks, then the priors' own model with Ramsey T2 (DD off) and
    with echo T2 everywhere T2 enters (DD on, idealized: the DD pulses themselves cost nothing), each decoded with
    its own priors and with the Ramsey ones."""
    cal = fz.cal
    out: dict[str, Any] = {"label": "SIMULATED", "calibration_set_id": cal.calibration_set_id, "shots": shots, "seed": seed,
                           "circuits": {}}
    for name in (*E1_CIRCUITS, *E4_CIRCUITS):
        spec = fz.suite[name]
        noiseless = spec.stim.compile_sampler(seed=seed).sample(256).astype(np.uint8)
        d0, o0 = D.detection_events(spec.stim, noiseless)
        rows = IR.detector_rows(spec, fz.suite.arch)
        t = np.array([r["round"] for r in rows])
        dec = {t2: decoder_priors(cal, spec, t2)[0] for t2 in ("ramsey", "echo")}
        dem = dec["ramsey"].detector_error_model(decompose_errors=True)
        c: dict[str, Any] = {
            "basis": spec.params["basis"], "rounds": spec.rounds, "delay_ns": spec.params.get("delay_ns", 0.0),
            "detectors": spec.stim.num_detectors, "observables": spec.stim.num_observables,
            "measurements": spec.stim.num_measurements,
            "noiseless": {"shots": 256, "detection_events": int(d0.sum()), "observable_flips": int(o0.sum())},
            "dem_errors": dem.num_errors, "graphlike_distance": len(dec["ramsey"].shortest_graphlike_error()),
        }
        for t2 in ("ramsey", "echo"):
            raw = dec[t2].compile_sampler(seed=seed).sample(shots).astype(np.uint8)
            dets, obs = D.detection_events(spec.stim, raw)
            k_own = int((D.decode(dec[t2], dets)[:, 0] != obs[:, 0]).sum())
            k_ram = int((D.decode(dec["ramsey"], dets)[:, 0] != obs[:, 0]).sum())
            c[t2] = {"detection_fraction": float(dets.mean()), "per_round": per_round(dets, t),
                     "p_l_own_priors": proportion(k_own, shots), "p_l_ramsey_priors": proportion(k_ram, shots),
                     "obs_flips": proportion(int(obs[:, 0].sum()), shots)}
        out["circuits"][name] = c
        log(f"{name}: noiseless {c['noiseless']}, DEM {c['dem_errors']} errors, distance {c['graphlike_distance']}; "
            f"P_L ramsey {c['ramsey']['p_l_own_priors']['p']:.4f}, echo {c['echo']['p_l_own_priors']['p']:.4f}; "
            f"detection {c['ramsey']['detection_fraction']:.4f} / {c['echo']['detection_fraction']:.4f}")
    return out


def cmd_predict(args: argparse.Namespace) -> int:
    fz = Frozen()
    res = predict(fz)
    for exp in ("E1", "E4"):
        body = {**res, "experiment": exp, "circuits": {k: v for k, v in res["circuits"].items() if k.startswith(exp)},
                "model": "dqiqm.experiments.device.hardware_noise(the committed snapshot, spec, t2=...): the decoder priors "
                         "as a Stim noise model; 'ramsey' stands for DD off, 'echo' for DD on (an upper bound on DD's gain: "
                         "the DD pulses are free in it). The hardware's detection fraction runs 1.3-1.75x the model's "
                         "(the batch), so these are a direction and a size, not a band.",
                "created_utc": utc_now(), "commit": hw.git_head()}
        write_json(RECORD / "ler" / exp / "dd_ab" / "prediction.json", body)
    bad = [k for k, v in res["circuits"].items() if v["noiseless"]["detection_events"] or v["noiseless"]["observable_flips"]
           or v["graphlike_distance"] != 3]
    log(f"PREDICTED ({len(res['circuits'])} circuits); noiseless or distance problems: {bad or 'none'}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# doc: the tables of docs/emerald-dd.md
# ---------------------------------------------------------------------------

DOC_BEGIN, DOC_END = "<!-- dd:begin -->", "<!-- dd:end -->"


def _ci(ci: Sequence[float] | None, nd: int = 4) -> str:
    return "" if not ci or ci[0] is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _pp(p: Mapping[str, Any]) -> str:
    return f"{p['k']}/{p['n']} = {p['p']:.4f} {_ci(p['wilson95'])}"


def _dd(d: Mapping[str, Any]) -> str:
    return f"{d['diff']:+.4f} {_ci(d['ci95'])} (z {d['z']:+.1f})"


def doc_tables(root: Path = RECORD) -> list[str]:
    lines: list[str] = []
    ab = {exp: json.loads((root / "ler" / exp / "dd_ab" / "ab.json").read_text())
          for exp in ("E1", "E4") if (root / "ler" / exp / "dd_ab" / "ab.json").exists()}
    pred = {exp: json.loads((root / "ler" / exp / "dd_ab" / "prediction.json").read_text())
            for exp in ("E1", "E4") if (root / "ler" / exp / "dd_ab" / "prediction.json").exists()}
    jobs = [j for exp in ab.values() for j in exp["jobs"]]
    seen = set()
    if jobs:
        lines += ["### The jobs (HARDWARE)", "",
                  "| job | DD | job id | calibration set | circuits x shots | ledger (UTC) | queue (s) | execution (s) | per shot (µs) | server dd_mode | checks |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for j in jobs:
            if j["job_id"] in seen:
                continue
            seen.add(j["job_id"])
            tl = j["timeline"]
            lines.append(f"| {j['label']} | {j['dd']} | `{j['job_id']}` | {j['calibration_set_id'][:8]} | {j['circuits']} x {j['shots']} "
                         f"| {j['ledger_time']} | {tl.get('queue_s', float('nan')):.1f} | {tl.get('execution_s', float('nan')):.1f} "
                         f"| {tl.get('execution_per_shot_us', float('nan')):.0f} | {j['server_dd_mode']} "
                         f"| {'OK' if j['checked_ok'] else 'NOT OK'} |")
        lines.append("")
    for exp, body in ab.items():
        if body.get("incomplete"):
            lines += [f"### {exp}: incomplete ({body})", ""]
            continue
        on, off = "+".join(body["on_jobs"]), "+".join(body["off_jobs"])
        for dec, title in (("ramsey", "decoder hardware_noise (Ramsey T2), both arms: the headline"),
                           ("echo", "decoder hardware_noise t2='echo', both arms"),
                           ("obs_flips", "undecoded: the observable's raw flip rate")):
            lines += [f"### {exp}, P_L, {title}", "",
                      f"| basis | R | DD on ({on}) | DD off ({off}) | off − on [Newcombe 95%] | "
                      + ("drift on A1 − A2 | drift off B1 − B2 |" if exp == "E1" else ""),
                      "|---|---|---|---|---|" + ("---|---|" if exp == "E1" else "")]
            for basis, by_r in body["cells"].items():
                for r, cell in sorted(by_r.items(), key=lambda kv: int(kv[0])):
                    c = cell[dec]
                    row = f"| {basis} | {r} | {_pp(c['on'])} | {_pp(c['off'])} | {_dd(c['off_minus_on'])} |"
                    if exp == "E1" and "drift_on" in c:
                        row += f" {_dd(c['drift_on'])} | {_dd(c['drift_off'])} |"
                    lines.append(row)
            lines.append("")
        lines += [f"### {exp}, detection fraction (decoder-free)", "",
                  "| basis | R | on | off | off − on | per job | per round, on | per round, off |", "|---|---|---|---|---|---|---|---|"]
        for basis, by_r in body["cells"].items():
            for r, cell in sorted(by_r.items(), key=lambda kv: int(kv[0])):
                d = cell["detection"]
                pr = lambda x: ", ".join(f"{v:.3f}" for _, v in sorted(x.items(), key=lambda kv: int(kv[0])))  # noqa: E731
                lines.append(f"| {basis} | {r} | {d['on']:.4f} | {d['off']:.4f} | {d['off_minus_on']:+.4f} | "
                             + ", ".join(f"{j} {v:.4f}" for j, v in d["per_job"].items())
                             + f" | {pr(d['per_round']['on'])} | {pr(d['per_round']['off'])} |")
        lines.append("")
        lines += [f"### {exp}, ε_L per round (2P = 1 − A(1 − 2ε)^R, maximum likelihood, bootstrap 95%)", "",
                  "| basis | decoder | ε on (A=1) | ε off (A=1) | ε off − on [bootstrap 95%] | ε on (free A), A | ε off (free A), A |",
                  "|---|---|---|---|---|---|---|"]
        for basis, e in body["eps"].items():
            for dec in ("ramsey", "echo"):
                fon, foff = e[f"on_{dec}"], e[f"off_{dec}"]
                diff = e[f"off_minus_on_{dec}"]
                free = lambda f: f"{f['free_A']['eps']:.4f}, {f['free_A']['amplitude']:.3f}" if "free_A" in f else ""  # noqa: E731
                lines.append(f"| {basis} | {dec} | {fon['fixed_A']['eps']:.4f} {_ci(fon['fixed_A']['eps_ci_bootstrap'])} "
                             f"| {foff['fixed_A']['eps']:.4f} {_ci(foff['fixed_A']['eps_ci_bootstrap'])} "
                             f"| {diff['diff']:+.4f} {_ci(diff['ci95_bootstrap'])} | {free(fon)} | {free(foff)} |")
            if "per_job_ramsey" in e:
                lines.append(f"| {basis} | ramsey, per job | "
                             + ", ".join(f"{j} {v['eps']:.4f}" for j, v in e["per_job_ramsey"].items())
                             + f" | | drift on {e['drift_on_ramsey']['diff']:+.4f} {_ci(e['drift_on_ramsey']['ci95_bootstrap'])}; "
                               f"drift off {e['drift_off_ramsey']['diff']:+.4f} {_ci(e['drift_off_ramsey']['ci95_bootstrap'])} | | |")
        lines.append("")
    for exp, show in (("E1", ("E1_d3_Z_r15_eA", "E1_d3_X_r15_eA")), ("E4", ("E4_d3_Z_r03_eA_t05000ns", "E4_d3_X_r03_eA_t05000ns"))):
        path = root / "ler" / exp / "dd_ab" / "by_stabilizer.json"
        if not path.exists():
            continue
        body = json.loads(path.read_text())
        on, off = "+".join(body["on_jobs"]), "+".join(body["off_jobs"])
        for name in show:
            c = body["circuits"][name]
            lines += [f"### {exp}, per stabilizer: `{name}` (decoder-free; data frames through the readout window)", "",
                      f"| stabilizer | ancilla | data (frame) | on ({on}) | off ({off}) | off − on | batch, DD off | "
                      "model Ramsey | model echo |", "|---|---|---|---|---|---|---|---|---|"]
            for s, e in c["stabilizers"].items():
                fr = ", ".join(f"{q} {'H' if f == 'ZYX' else 'I' if f == 'XYZ' else f}" for q, f in e["data_frames"].items())
                b = "" if e["batch_dd_off"] is None else f"{e['batch_dd_off']:.3f}"
                lines.append(f"| {s} | {e['ancilla']} | {fr} | {e['on']['rate']:.3f} | {e['off']['rate']:.3f} | "
                             f"{e['off_minus_on']:+.3f} | {b} | {e['model_ramsey']:.3f} | {e['model_echo']:.3f} |")
            lines += ["", f"Batch jobs: {', '.join(f'{k} `{v}`' for k, v in (c['batch_jobs'] or {}).items())}. "
                          "Frame I: the identity; H: a Hadamard frame, where a physical Z acts as Stim's X.", ""]
        lines += [f"### {exp}, deterministic first readouts: misread rate, off − on [Newcombe 95%]", "",
                  "| circuit | " + " | ".join(["ancilla: on / off / off − on"] * 4) + " |", "|---|---|---|---|---|"]
        for name, c in body["circuits"].items():
            cells = [f"{q}: {v['on']['p']:.3f} / {v['off']['p']:.3f} / {v['off_minus_on']['diff']:+.3f} {_ci(v['off_minus_on']['ci95'], 3)}"
                     for q, v in c["first_readout_misread"].items()]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    for exp, body in pred.items():
        lines += [f"### {exp}: what the priors' own model predicts (SIMULATED, {body['shots']:,} shots per circuit, seed {body['seed']})", "",
                  "| circuit | noiseless events / flips | DEM errors | distance | P_L Ramsey (DD off) | P_L echo (DD on, ideal) | detection Ramsey / echo |",
                  "|---|---|---|---|---|---|---|"]
        for name, c in body["circuits"].items():
            lines.append(f"| {name} | {c['noiseless']['detection_events']} / {c['noiseless']['observable_flips']} | {c['dem_errors']} "
                         f"| {c['graphlike_distance']} | {_pp(c['ramsey']['p_l_own_priors'])} | {_pp(c['echo']['p_l_own_priors'])} "
                         f"| {c['ramsey']['detection_fraction']:.4f} / {c['echo']['detection_fraction']:.4f} |")
        lines.append("")
    return lines


def cmd_doc(args: argparse.Namespace) -> int:
    text = DOC.read_text()
    if DOC_BEGIN not in text or DOC_END not in text:
        raise SystemExit(f"{DOC} has no {DOC_BEGIN} ... {DOC_END} block")
    head, rest = text.split(DOC_BEGIN, 1)
    _, tail = rest.split(DOC_END, 1)
    DOC.write_text(head + DOC_BEGIN + "\n" + "\n".join(doc_tables()) + "\n" + DOC_END + tail)
    log(f"wrote the tables of {_rel(DOC, REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="offline: the segments against the brief, and the arming lines")
    sub.add_parser("predict", help="offline (sim lock): the circuits' noiseless and DEM checks, and Stim's DD off/on prediction")
    sub.add_parser("snapshot", help="live, read-only: one Emerald snapshot")
    v = sub.add_parser("verify", help="the freeze, the live checks; --live also builds every run request read-only")
    v.add_argument("--segment", choices=ORDER, required=True)
    v.add_argument("--snapshot", type=Path)
    v.add_argument("--live", action="store_true")
    s = sub.add_parser("submit", help="the segment's jobs (armed by DQ_QPU=run ...)")
    s.add_argument("--segment", choices=ORDER, required=True)
    s.add_argument("--snapshot", type=Path, required=True, help="the before snapshot (its calibration set is used)")
    s.add_argument("--timeout", type=float, default=None, help="seconds to wait for each result (default: iqm-client's)")
    r = sub.add_parser("recover", help="read-only: re-pull a submitted job and save it as save_raw does")
    r.add_argument("--segment", choices=ORDER, required=True)
    r.add_argument("--job-id", required=True)
    r.add_argument("--snapshot", type=Path, help="the snapshot of the job's calibration set (for the sanity references)")
    r.add_argument("--timeout", type=float, default=None)
    a = sub.add_parser("analyze", help="offline: the records, the LER records and the A/B")
    a.add_argument("--snapshots", type=Path, nargs="+", required=True, help="before ... after (every calibration set the jobs ran on)")
    a.add_argument("--root", type=Path, default=RECORD)
    a.add_argument("--ledger", type=Path, default=Path(hw.DEFAULT_LEDGER))
    st = sub.add_parser("stabilizers", help="offline: per-stabilizer detection rates and first-readout misreads, on against off")
    st.add_argument("--root", type=Path, default=RECORD)
    sub.add_parser("doc", help="offline: rewrite docs/emerald-dd.md's tables from the records")
    args = ap.parse_args()
    return {"plan": cmd_plan, "predict": cmd_predict, "snapshot": cmd_snapshot, "verify": cmd_verify, "submit": cmd_submit,
            "recover": cmd_recover, "analyze": cmd_analyze, "stabilizers": cmd_stabilizers, "doc": cmd_doc}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
