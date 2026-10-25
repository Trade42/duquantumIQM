"""emerald-pilot: the first Emerald job. EP0, ``E1_d3_Z_r03_eA``, 1 circuit x 1,000 shots on IQM Emerald.

    uv run python scripts/emerald_pilot.py snapshot                        # live, read-only: resonance_check.py + patch check
    uv run python scripts/emerald_pilot.py verify [--snapshot S] [--live]  # offline; --live also builds the run request
    DQ_QPU=run DQ_QPU_TASK=emerald-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000 \\
        uv run python scripts/emerald_pilot.py submit --snapshot S         # the one job
    uv run python scripts/emerald_pilot.py recover --job-id J              # read-only; only if submit failed after the job id
    uv run python scripts/emerald_pilot.py analyze --before S --after T    # offline: stop/go, timeline, decode, InfoRecord

This is ``scripts/garnet_pilot.py`` (garnet-pilot, commit 26cd1f0) copied and adapted for Emerald;
``garnet_pilot.py`` itself is unchanged. Its module globals (task, device, arming, paths) are read by
almost every function, so the Emerald driver is a copy, not an import. What differs:

The freeze rule (emerald-design review, MINOR 1)
    The circuit is frozen: EP0 is always rebuilt from the **committed** snapshot,
    ``emerald.build(emerald.live_calibration(emerald.DEFAULT_SNAPSHOT))`` (calibration set
    23e12f6b), and its ``native_sha256`` must equal ``InfoRecord/emerald/circuits/E1/placement.json``'s.
    A live calibration never rebuilds it. ``verify --snapshot <live>`` checks the frozen EP0 against
    the live calibration instead: ``prx`` and ``measure`` served on its 17 qubits and ``cz`` on its
    24 couplers (``patch_gates``), ``emerald.check_live_couplers`` with the live calibration, and,
    with ``--live``, IQM's own ``create_run_request`` validation against the live architecture. When
    the live calibration set is not 23e12f6b, it also re-ranks the placements under the live noise
    (``memory.rank``) and prints where the frozen eA stands: for information only.
    ``InfoRecord/emerald/circuits/`` is never regenerated.

The qubits (as ``garnet_pilot.py`` established them from iqm-client 35.0.3's code)
    ``hw.submit`` calls ``IQMBackend.create_run_request(circuits, shots, qubit_index_to_name=...)``,
    which only serializes (no transpile, no layout pass): each instruction's qubit goes through the
    circuit's layout to its physical index, then through ``qubit_index_to_name``, the map EP0 was
    placed with, to a device name. ``IQMClient.create_run_request`` then validates the named loci
    against the calibration's dynamic architecture and sets ``qubit_mapping=None``. ``verify --live``
    builds that very request on the read-only ``hw.backend`` (its ``run`` and ``submit_*`` refuse) and
    checks that its instructions hash to the planned ``native_sha256`` and that ``hw._sent_qubits`` is
    exactly eA's 17 qubits.

``submit`` refuses unless it is armed exactly for this task (1 job, 1,000 shots, the default ledger),
no ``emerald-pilot`` row is in the ledger, ``InfoRecord/emerald/raw/*-emerald-pilot/`` holds no job,
the frozen checks pass against the before snapshot, and the server's default calibration set is still
the snapshot's. Then it calls ``hw.submit`` once, with the snapshot's calibration set id, and prints
the job id the moment ``hw.submit`` writes the ledger row (a read-only watch on the ledger), before
the result. If anything fails after that, ``recover`` re-pulls the job read-only with ``hw.retrieve``
and saves it with ``hw.save_raw`` into the same layout. It never submits.

``analyze`` reads the raw job and runs the stop/go checks of ``job_plan.json``'s ``pilot.band``, plus
the emerald-design review's MINOR 3: ``meta.json``'s ``physical_qubits`` and the job's ledger row
name exactly eA's 17 qubits. It reads the timeline from ``meta.json``, decodes with PyMatching on the
noisy copy of the circuit (priors ``emerald.calibrated_noise(<before snapshot>, spec,
idle_ns=emerald.readout_window_ns(...))``, 360 ns), samples that same model (100,000 shots, seed
2026) for the per-round comparison, and writes ``InfoRecord/emerald/datasets/E1/ep0_E1_d3_Z_r03_eA/``,
``derived/`` (plus ``pilot_checks.json``) and ``ler/``, labelled HARDWARE with ``device: emerald``.
Nothing here needs the sim lock: the suite builds in about 25 s of CPU, and decoding is instant.
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
from collections import Counter
from collections.abc import Mapping, Sequence
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
from dqiqm.experiments import memory
from dqiqm.experiments import suite as SU

REPO = Path(__file__).resolve().parents[1]
RECORD = EM.RECORD
RAW_ROOT = RECORD / "raw"
CALIBRATION = REPO / "data" / "calibration"
PLACEMENT = RECORD / "circuits" / "E1" / "placement.json"
JOB_PLAN = RECORD / "circuits" / "job_plan.json"
#: Garnet's pilot, for the ratios the brief asks to compare against.
GARNET_PILOT_CHECKS = REPO / "InfoRecord" / "derived" / "E1" / "p0_E1_d3_Z_r03_pA" / "pilot_checks.json"

TASK = "emerald-pilot"
DEVICE = EM.DEVICE
CIRCUIT = "E1_d3_Z_r03_eA"
SHOTS = 1000
JOB_LABEL = "EP0"
RUN_ID = "ep0_" + CIRCUIT
PURPOSE = f"emerald-pilot EP0 {CIRCUIT}"
DESIGN_CALSET = EM.CALSET
#: Patch eA (the brief): QB13, QB14, QB19-QB22, QB27-QB31, QB36-QB39, QB44, QB45.
PATCH_QUBITS = tuple(f"QB{i}" for i in (13, 14, *range(19, 23), *range(27, 32), *range(36, 40), 44, 45))
NATIVE_OPS = {"prx", "cz", "measure", "barrier"}
#: Exactly the arming the brief grants, and nothing wider.
ARMING = {"DQ_QPU": "run", "DQ_QPU_TASK": TASK, "DQ_QPU_MAX_JOBS": "1", "DQ_QPU_MAX_SHOTS": str(SHOTS)}
#: The calibrated model's own sample for the per-round comparison (job_plan.json's pilot block uses the same).
MODEL_SHOTS, MODEL_SEED = 100_000, 2026


def log(msg: str = "") -> None:
    print(msg, flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# The patch: its qubits and couplers, from the committed placement.
# ---------------------------------------------------------------------------


def placement_entry() -> dict[str, Any]:
    return json.loads(PLACEMENT.read_text())[CIRCUIT]


def patch_from_placement(entry: Mapping[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    """The 17 device qubits and the 24 couplers (one per ancilla-data CX) of EP0's placement."""
    phys = entry["placement"]
    qubits = sorted(set(phys.values()), key=dev.qubit_sort_key)
    couplers = set()
    for anc, st in entry["stabilizers"].items():
        for d in st["support"]:
            if entry["roles"][str(d)] == "data":
                couplers.add(dev.pair_key(phys[anc], phys[str(d)]))
    return qubits, sorted(couplers, key=lambda p: (dev.qubit_sort_key(p[0]), dev.qubit_sort_key(p[1])))


def _served(arch: Mapping[str, Any], gate: str, locus: tuple[str, ...]) -> bool:
    """Whether the implementation that runs ``gate`` on ``locus`` (its override, else the default) lists it."""
    info = arch["gates"].get(gate)
    if not info:
        return False
    impl = dev._impl(info, locus)  # noqa: SLF001 -- the rule calibration_from_snapshot reads the figures by
    loci = info["implementations"].get(impl, {}).get("loci", [])
    return any(len(lc) == len(locus) and set(lc) == set(locus) for lc in loci)


def patch_gates(arch: Mapping[str, Any], qubits: Sequence[str], couplers: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """What a dynamic architecture is missing for the patch: prx and measure per qubit, cz per coupler."""
    return {
        "prx": [q for q in qubits if not _served(arch, "prx", (q,))],
        "measure": [q for q in qubits if not _served(arch, "measure", (q,))],
        "cz": [f"{a}-{b}" for a, b in couplers if not _served(arch, "cz", (a, b))],
    }


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


def _snapshot_files() -> set[Path]:
    return {p for d in ("garnet", "emerald") for p in (CALIBRATION / d).glob("*.json")}


def load_emerald_snapshot(path: Path) -> dict[str, Any]:
    snap = json.loads(Path(path).read_text())
    if snap.get("device") != DEVICE:
        raise SystemExit(f"{path}: device {snap.get('device')!r}, not {DEVICE}")
    return snap


def cmd_snapshot(args: argparse.Namespace) -> int:
    before = _snapshot_files()
    log(f"{utc_now()} running scripts/resonance_check.py live (read-only)")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "resonance_check.py")], cwd=REPO)
    new = sorted(_snapshot_files() - before)
    for p in new:
        log(f"new snapshot: {p.relative_to(REPO)}")
    emerald = [p for p in new if p.parent.name == DEVICE]
    if proc.returncode != 0 or len(emerald) != 1:
        log(f"STOP: resonance_check.py exited {proc.returncode}, {len(emerald)} new emerald snapshot(s)")
        return 2
    snap = load_emerald_snapshot(emerald[0])
    calset = snap["calibration_set_id"]
    log(f"SNAPSHOT {emerald[0].relative_to(REPO)}")
    log(f"  calibration_set_id {calset}; server default at fetch {snap.get('default_calibration_set_id_at_fetch')}")
    log(f"  fetched {snap['fetched_utc_start']} .. {snap['fetched_utc_end']}")
    log(f"  same as the design's {DESIGN_CALSET}: {calset == DESIGN_CALSET}")
    qubits, couplers = patch_from_placement(placement_entry())
    missing = patch_gates(snap["architecture"], qubits, couplers)
    log(f"  patch eA: {len(qubits)} qubits, {len(couplers)} couplers; missing {missing}")
    if any(missing.values()):
        log("STOP: the live calibration lacks a gate the frozen EP0 needs")
        return 3
    log("  PATCH OK: prx and measure on all 17 qubits, cz on all 24 couplers")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def instructions_json(instructions: Sequence[Any]) -> list[dict[str, Any]]:
    """``circuits.native_json`` for any sequence of serialized instructions (a run request's too)."""
    return [{"name": i.name, "locus": list(i.locus),
             "args": {k: (float(v) if isinstance(v, float) else v) for k, v in i.args.items()}} for i in instructions]


def instructions_sha256(instructions: Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps(instructions_json(instructions), sort_keys=True).encode()).hexdigest()


class Pilot:
    """EP0 rebuilt from the committed snapshot (frozen), and everything ``submit`` and ``analyze`` need from it."""

    def __init__(self) -> None:
        self.cal = EM.live_calibration(EM.DEFAULT_SNAPSHOT)
        self.suite = EM.build(self.cal)
        assert EM.pilot_name(self.suite) == CIRCUIT, EM.pilot_name(self.suite)
        self.spec = self.suite[CIRCUIT]
        self.hc = self.spec.hardware
        self.entry = placement_entry()
        self.qubit_index_to_name = dict(self.hc.qubit_index_to_name)
        self.qubits, self.couplers = patch_from_placement(self.entry)
        # what hw.submit will record in meta.json (qubit_mapping) and the ledger (physical qubits)
        self.qubit_mapping = hw._qubit_mappings(None, [self.hc.circuit], self.qubit_index_to_name)  # noqa: SLF001
        self.sent = [sorted({q for i in C.sent(self.hc) for q in i.locus}, key=dev.qubit_sort_key)]


def offline_checks(p: Pilot) -> dict[str, Any]:
    """The frozen EP0 against the committed record. Raise ``AssertionError`` on the first failure; return the numbers."""
    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey

    hc, spec, entry = p.hc, p.spec, p.entry
    assert p.cal.calibration_set_id == DESIGN_CALSET, f"the committed snapshot is {p.cal.calibration_set_id}"
    assert hc.target == "hardware" and entry["target"] == "hardware", hc.target
    assert entry["device"] == DEVICE and spec.params["patch"] == "eA", (entry["device"], spec.params)
    native = C.native_sha256(hc)
    assert native == entry["native_sha256"], f"FREEZE: native_sha256 {native} != placement.json {entry['native_sha256']}"
    assert spec.stim_sha256() == entry["stim_sha256"], "stim_sha256 differs from placement.json"
    qasm_sha = hashlib.sha256((C.qasm(hc) + "\n").encode()).hexdigest()
    assert qasm_sha == entry["qasm_sha256"], "qasm_sha256 differs from placement.json"

    qiskit_ops = dict(hc.circuit.count_ops())
    assert set(qiskit_ops) <= {"r", "cz", "measure", "barrier"}, qiskit_ops
    assert "swap" not in qiskit_ops
    sent = C.sent(hc)
    ops = Counter(i.name for i in sent)
    assert set(ops) <= NATIVE_OPS, f"non-native ops {set(ops) - NATIVE_OPS}"
    assert ops["measure"] == 33 == spec.stim.num_measurements, ops
    # every cz on one of the committed calibration's 85 CZ couplers, none on the 5 CZ-less ones
    live_couplers = EM.check_live_couplers(spec, p.cal)
    calibrated = {frozenset(c) for c in p.cal.couplers}
    dead = {frozenset(c) for c in EM.CZ_LESS}
    cz = [frozenset(i.locus) for i in sent if i.name == "cz"]
    assert all(c in calibrated and c not in dead for c in cz), "a cz off the calibrated couplers"
    assert {frozenset(c) for c in p.couplers} == set(cz), "the cz loci are not the patch's 24 couplers"
    assert len(set(cz)) == 24 and len(cz) == 72 == live_couplers["cz"], (len(set(cz)), len(cz))
    # measurement j -> clbit j, on the device qubit its Stim qubit is placed on
    meas_q = C.measured_qubits(spec.stim)
    meas = [i for i in sent if i.name == "measure"]
    for inst in meas:
        mk = MeasurementKey.from_string(inst.args["key"])
        assert mk.creg_idx == 0 and mk.creg_len == 33, mk
        (qb,) = inst.locus
        assert qb == spec.physical[meas_q[mk.clbit_idx]], f"clbit {mk.clbit_idx} measured on {qb}"
    assert sorted(MeasurementKey.from_string(i.args["key"]).clbit_idx for i in meas) == list(range(33))

    qubits = sorted({q for i in sent for q in i.locus}, key=dev.qubit_sort_key)
    assert qubits == p.qubits == sorted(entry["placement"].values(), key=dev.qubit_sort_key) == list(PATCH_QUBITS), qubits
    assert sorted(p.qubit_index_to_name.values(), key=dev.qubit_sort_key) == list(PATCH_QUBITS)
    (mapping,) = p.qubit_mapping
    assert mapping == {v: spec.physical[s] for v, s in enumerate(hc.stim_qubits)}, "qubit_mapping != the placement"
    assert p.sent == [list(PATCH_QUBITS)]
    return {
        "native_sha256": native,
        "stim_sha256": entry["stim_sha256"],
        "qasm_sha256": qasm_sha,
        "qiskit_ops": qiskit_ops,
        "sent_ops": dict(ops),
        "cz": len(cz),
        "couplers": len(set(cz)),
        "cz_less_used": 0,
        "measurements": len(meas),
        "swaps": 0,
        "qubits": qubits,
        "qubit_index_to_name": {str(k): v for k, v in sorted(p.qubit_index_to_name.items())},
        "qubit_mapping": {str(k): v for k, v in sorted(mapping.items())},
    }


def live_checks(p: Pilot, snapshot: Path) -> dict[str, Any]:
    """The frozen EP0 against a live snapshot's calibration: nothing rebuilt. Raises ``AssertionError``."""
    snap = load_emerald_snapshot(snapshot)
    missing = patch_gates(snap["architecture"], p.qubits, p.couplers)
    assert not any(missing.values()), f"STOP: the live calibration lost a gate the frozen EP0 needs: {missing}"
    live_cal = EM.live_calibration(snapshot)
    couplers = EM.check_live_couplers(p.spec, live_cal)
    return {"snapshot": _rel(Path(snapshot), REPO), "calibration_set_id": snap["calibration_set_id"],
            "same_as_design": snap["calibration_set_id"] == DESIGN_CALSET, "missing": missing,
            "check_live_couplers": couplers}


def rerank_info(p: Pilot, snapshot: Path, shots: int = 50_000) -> dict[str, Any]:
    """For information only: the placements re-ranked under a live calibration, and where the frozen eA stands."""
    live_cal = EM.live_calibration(snapshot)
    ranked = memory.rank(EM.architecture(live_cal), live_cal, shots=shots)
    by_set = EM.ranking_by_set(ranked)
    ea = frozenset(PATCH_QUBITS)
    frozen = next((i for i, r in enumerate(ranked) if EM.qubit_set(r) == ea
                   and r["placement"].orientation == p.entry["orientation"]
                   and list(r["placement"].origin) == list(p.entry["origin"])), None)
    top = [{"rank": i, "orientation": r["placement"].orientation, "origin": list(r["placement"].origin),
            "eA_set": EM.qubit_set(r) == ea, "mean_p_l_r3": r["mean_p_l"]} for i, r in enumerate(ranked[:6])]
    return {"calibration_set_id": live_cal.calibration_set_id, "rank_shots": shots, "placements": len(ranked),
            "frozen_eA_rank": frozen, "frozen_eA_mean_p_l_r3": None if frozen is None else ranked[frozen]["mean_p_l"],
            "eA_set_rank_by_set": next((i for i, r in enumerate(by_set) if EM.qubit_set(r) == ea), None),
            "top": top, "note": "information only: the circuits stay frozen (emerald-design review, MINOR 1)"}


def run_request_checks(rr: Any, p: Pilot, calset: str) -> dict[str, Any]:
    """The run request ``hw.submit`` would build, against the plan. Raises ``AssertionError``."""
    from iqm.iqm_client.models import DDMode, HeraldingMode

    assert len(rr.circuits) == 1
    sha = instructions_sha256(rr.circuits[0].instructions)
    assert sha == p.entry["native_sha256"], f"the run request's instructions hash to {sha}"
    sent = hw._sent_qubits(rr)  # noqa: SLF001 -- what the ledger row will record
    assert sent == [list(PATCH_QUBITS)], sent
    assert rr.qubit_mapping is None, rr.qubit_mapping
    assert str(rr.calibration_set_id) == calset, rr.calibration_set_id
    assert rr.shots == SHOTS, rr.shots
    assert rr.heralding_mode == HeraldingMode.NONE and rr.dd_mode == DDMode.DISABLED
    assert rr.active_reset_cycles is None
    return {
        "instructions_sha256": sha,
        "sent_qubits": sent[0],
        "qubit_mapping": rr.qubit_mapping,
        "calibration_set_id": str(rr.calibration_set_id),
        "shots": rr.shots,
        "options": {k: str(getattr(rr, k)) for k in ("heralding_mode", "dd_mode", "dd_strategy", "active_reset_cycles",
                                                      "max_circuit_duration_over_t2", "move_gate_validation",
                                                      "move_gate_frame_tracking")},
    }


def live_run_request(p: Pilot, calset: str) -> Any:
    """``create_run_request`` exactly as ``hw.submit`` calls it, on the read-only backend (it cannot submit)."""
    b = hw.backend(DEVICE, calset)
    assert str(b.architecture.calibration_set_id) == calset, b.architecture.calibration_set_id
    return b.create_run_request([p.hc.circuit], shots=SHOTS, qubit_index_to_name=p.qubit_index_to_name)


def allowance_check(ledger: Path) -> dict[str, Any]:
    """``hw.check_allowance`` exactly as the armed ``submit`` will run it, without arming. Raises ``QPUGuardError``."""
    env = {**ARMING, "DQ_QPU_LEDGER": str(ledger)}
    hw.check_allowance(TASK, 1, SHOTS, env=env)
    rows = hw.read_ledger(ledger)
    return {"ledger": str(ledger), "rows": len(rows), "task_rows": sum(r.task == TASK for r in rows),
            "task_usage": list(hw.usage(TASK, ledger)), "fits": True}


def cmd_verify(args: argparse.Namespace) -> int:
    p = Pilot()
    out: dict[str, Any] = {"offline": offline_checks(p)}
    log("OFFLINE OK (frozen EP0 = placement.json): " + json.dumps(out["offline"]))
    if args.snapshot:
        out["live"] = live_checks(p, args.snapshot)
        log(f"LIVE CALIBRATION OK in {args.snapshot}: " + json.dumps(out["live"]))
        if not out["live"]["same_as_design"]:
            out["rerank"] = rerank_info(p, args.snapshot)
            log("RE-RANK (information only, nothing rebuilt): " + json.dumps(out["rerank"]))
        out["allowance"] = allowance_check(hw.ledger_path())
        log("ALLOWANCE OK (not armed; the check submit will run): " + json.dumps(out["allowance"]))
        if args.live:
            calset = out["live"]["calibration_set_id"]
            out["run_request"] = run_request_checks(live_run_request(p, calset), p, calset)
            log("RUN REQUEST OK (read-only backend, not submitted): " + json.dumps(out["run_request"]))
    elif args.live:
        raise SystemExit("--live needs --snapshot (the calibration set to build the request for)")
    return 0


# ---------------------------------------------------------------------------
# submit and recover
# ---------------------------------------------------------------------------


def check_arming(env: Mapping[str, str]) -> None:
    """Refuse unless armed exactly as the brief grants, against the default ledger."""
    for k, v in ARMING.items():
        if env.get(k) != v:
            raise SystemExit(f"refusing: {k}={env.get(k)!r}, the brief arms {k}={v}")
    if hw.ledger_path(env) != Path(hw.DEFAULT_LEDGER):
        raise SystemExit(f"refusing: ledger {hw.ledger_path(env)}, not {hw.DEFAULT_LEDGER}")


def existing_jobs(raw_root: Path = RAW_ROOT) -> list[Path]:
    """Anything already under ``<raw_root>/*-emerald-pilot/``."""
    return sorted(p for d in Path(raw_root).glob(f"*-{TASK}") for p in d.iterdir())


def ledger_rows(ledger: Path, task: str | None = TASK) -> list[hw.LedgerRow]:
    """The ledger's rows of ``task`` (every row when ``task`` is None)."""
    if not Path(ledger).is_file():
        return []
    return [r for r in hw.read_ledger(ledger) if task is None or r.task == task]


class LedgerWatch:
    """Print the task's job id as soon as ``hw.submit`` writes its ledger row. Reads the ledger only."""

    def __init__(self, ledger: Path, task: str = TASK, interval: float = 0.5) -> None:
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


def check_ledger_row(row: hw.LedgerRow, *, job_id: str, calset: str, qubits: Sequence[str]) -> dict[str, Any]:
    want = {"task": TASK, "device": DEVICE, "calibration_set_id": calset, "qubits": ",".join(qubits),
            "circuits x shots": f"1 x {SHOTS}", "job id": job_id}
    got = {"task": row.task, "device": row.device, "calibration_set_id": row.calibration_set_id, "qubits": row.qubits,
           "circuits x shots": f"{row.circuits} x {row.shots}", "job id": row.job_id}
    return {k: {"want": want[k], "got": got[k], "ok": want[k] == got[k]} for k in want}


def submit_pilot(p: Pilot, *, calset: str, ledger: Path, data_root: Path, timeout: float | None) -> dict[str, Any]:
    """The one ``hw.submit``. Returns what happened; on a failure after the job id, says how to recover."""
    if ledger_rows(ledger):
        raise SystemExit(f"refusing: {ledger} already has a {TASK} row; there is no resubmission")
    if existing_jobs(data_root):
        raise SystemExit(f"refusing: {data_root}/*-{TASK}/ already holds {existing_jobs(data_root)}")
    error = None
    with LedgerWatch(ledger):  # prints the job id the moment the row lands
        try:
            sub = hw.submit([p.hc.circuit], device=DEVICE, shots=SHOTS, purpose=PURPOSE, calibration_set_id=calset,
                            qubit_index_to_name=p.qubit_index_to_name, data_root=data_root, timeout=timeout)
        except BaseException as e:  # noqa: BLE001 -- reported below, whatever it is
            error, sub = e, None
    rows = ledger_rows(ledger)
    job_id = rows[0].job_id if rows else None
    log(f"JOB ID {job_id}" if job_id else "NO JOB ID: no ledger row was written, so no job was created")
    if error is not None:
        log(f"hw.submit raised {type(error).__name__}: {error}")
        if job_id:
            log(f"RECOVER (read-only, never submit again): uv run python scripts/emerald_pilot.py recover --job-id {job_id}")
        return {"job_id": job_id, "error": f"{type(error).__name__}: {error}", "ok": False}
    row_check = check_ledger_row(rows[0], job_id=sub.job_id, calset=calset, qubits=PATCH_QUBITS)
    meta = json.loads((sub.path / "meta.json").read_text())
    meta_check = {
        "qubit_mapping": meta["qubit_mapping"] == [{str(k): v for k, v in sorted(m.items())} for m in p.qubit_mapping],
        "physical_qubits": meta["physical_qubits"] == [list(PATCH_QUBITS)],
        "calibration_set_id": meta["calibration_set_id"] == calset,
        "shots": meta["shots"] == SHOTS,
        "success": meta["success"],
        "timeline": bool(meta.get("timeline")),
    }
    ok = len(rows) == 1 and all(v["ok"] for v in row_check.values()) and all(meta_check.values())
    log(f"raw: {sub.path}")
    log("ledger row: " + json.dumps(row_check))
    log(f"ledger: {len(ledger_rows(ledger, None))} rows in all, {len(rows)} {TASK}")
    log("meta.json: " + json.dumps(meta_check))
    return {"job_id": sub.job_id, "path": str(sub.path), "calibration_set_id": sub.calibration_set_id,
            "qubits": sub.qubits, "ledger_rows": len(rows), "ledger_rows_all": len(ledger_rows(ledger, None)),
            "ledger_check": row_check, "meta_check": meta_check, "ok": ok}


def cmd_submit(args: argparse.Namespace) -> int:
    check_arming(os.environ)
    ledger = hw.ledger_path()
    snap = load_emerald_snapshot(args.snapshot)
    calset = snap["calibration_set_id"]
    p = Pilot()
    log("OFFLINE OK (frozen EP0 = placement.json): " + json.dumps(offline_checks(p)))
    try:
        live = live_checks(p, args.snapshot)
    except AssertionError as e:
        raise SystemExit(f"refusing: {e}") from None
    log("LIVE CALIBRATION OK: " + json.dumps(live))
    log("RUN REQUEST OK (read-only backend, not submitted): "
        + json.dumps(run_request_checks(live_run_request(p, calset), p, calset)))
    default_now = hw.calibration_set_id(DEVICE)
    if default_now != calset:
        raise SystemExit(f"refusing: the server's default calibration set is now {default_now}, the snapshot's {calset}; "
                         "take a new snapshot")
    log(f"{utc_now()} submitting {CIRCUIT}: 1 circuit x {SHOTS} shots on {DEVICE}, calibration set {calset} "
        f"(the before snapshot's, = the server default), qubits {','.join(PATCH_QUBITS)}")
    res = submit_pilot(p, calset=calset, ledger=ledger, data_root=RAW_ROOT, timeout=args.timeout)
    log("SUBMIT " + ("OK" if res["ok"] else "NOT OK") + ": " + json.dumps(res))
    return 0 if res["ok"] else 4


def recover_pilot(p: Pilot, job_id: str, *, ledger: Path, data_root: Path, timeout: float | None) -> Path:
    """Re-pull the job read-only and save it in ``save_raw``'s layout, with the submission's qubits."""
    rows = [r for r in ledger_rows(ledger) if r.job_id == job_id]
    if len(rows) != 1:
        raise SystemExit(f"refusing: job {job_id} has {len(rows)} {TASK} rows in {ledger}")
    day = rows[0].time[:10].replace("-", "")
    target = Path(data_root) / f"{day}-{TASK}" / job_id
    if target.exists():
        raise SystemExit(f"refusing: {target} exists ({sorted(x.name for x in target.iterdir())}); it is never overwritten")
    result = hw.retrieve(job_id, DEVICE, timeout=timeout)
    return hw.save_raw(result, task=TASK, device=DEVICE, qubit_mapping=p.qubit_mapping, shots=SHOTS,
                       physical_qubits=p.sent, purpose=PURPOSE, day=day, data_root=data_root)


def cmd_recover(args: argparse.Namespace) -> int:
    p = Pilot()
    offline_checks(p)
    path = recover_pilot(p, args.job_id, ledger=hw.ledger_path(), data_root=RAW_ROOT, timeout=args.timeout)
    log(f"RECOVERED (hw.retrieve + hw.save_raw, by hand): {path}")
    return 0


# ---------------------------------------------------------------------------
# analyze: stop/go, timeline, decode, InfoRecord
# ---------------------------------------------------------------------------


def pilot_band(plan_pilot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``job_plan.json``'s ``pilot.band`` with the pilot block's ``x_cols`` and ``z_cols`` (Emerald keeps them outside the band)."""
    plan_pilot = plan_pilot or json.loads(JOB_PLAN.read_text())["pilot"]
    return {**plan_pilot["band"], "x_cols": list(plan_pilot["x_cols"]), "z_cols": list(plan_pilot["z_cols"])}


def stop_go(raw: np.ndarray, dets: np.ndarray, band: Mapping[str, Any], shots: int = SHOTS) -> list[dict[str, Any]]:
    """The pilot's checks (``job_plan.json: pilot.band``), each with its value, band and pass."""
    raw = np.asarray(raw)
    cols = int(band["columns"])
    checks = []

    def add(name: str, value: Any, want: Any, ok: bool) -> None:
        checks.append({"check": name, "value": value, "band": want, "pass": bool(ok)})

    shaped = raw.ndim == 2 and raw.shape == (shots, cols)
    add("memory shape", list(raw.shape), [shots, cols], shaped)
    if not shaped:
        for name in ("all-zero rows", "X-type round-1 bits, pooled mean", "Z-type round-1 bits, pooled mean",
                     "detection fraction"):
            add(name, None, "not computed: wrong shape", False)
        return checks
    max_zero = band["all_zero_rows_max_per_1000"] * shots / 1000
    zeros = int((~raw.any(axis=1)).sum())
    add("all-zero rows", zeros, f"<= {max_zero:g}", zeros <= max_zero)
    lo, hi = band["x_ancilla_m1_one"]
    x = float(raw[:, band["x_cols"]].mean())
    add("X-type round-1 bits, pooled mean", x, [lo, hi], lo <= x <= hi)
    z = float(raw[:, band["z_cols"]].mean())
    add("Z-type round-1 bits, pooled mean", z, f"< {band['z_ancilla_m1_one_max']}", z < band["z_ancilla_m1_one_max"])
    lo, hi = band["detection_fraction"]
    f = float(np.asarray(dets).mean())
    add("detection fraction", f, [lo, hi], lo <= f <= hi)
    return checks


def qubit_checks(meta: Mapping[str, Any], rows: Sequence[hw.LedgerRow] | None, *, calset: str,
                 want_mapping: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Where the job ran (emerald-design review, MINOR 3): ``meta.json`` and the ledger row name exactly eA's 17 qubits.

    ``rows`` are the ledger's rows with this job's id, of any task (None: no ledger, a simulated rehearsal).
    """
    want = [list(PATCH_QUBITS)]
    checks = [{"check": "physical qubits (meta.json) = eA's 17", "value": meta.get("physical_qubits"), "band": want,
               "pass": meta.get("physical_qubits") == want},
              {"check": "qubit_mapping (meta.json) = the placement", "value": meta.get("qubit_mapping"),
               "band": list(want_mapping), "pass": meta.get("qubit_mapping") == list(want_mapping)}]
    if rows is None:
        checks.append({"check": "ledger row = eA's 17, 1 x 1000", "value": None, "band": "not applicable: simulated",
                       "pass": True})
        return checks
    if len(rows) != 1:
        checks.append({"check": "ledger row = eA's 17, 1 x 1000", "value": f"{len(rows)} rows with this job id",
                       "band": "exactly 1", "pass": False})
        return checks
    rc = check_ledger_row(rows[0], job_id=meta["job_id"], calset=calset, qubits=PATCH_QUBITS)
    checks.append({"check": "ledger row = eA's 17, 1 x 1000", "value": {k: v["got"] for k, v in rc.items()},
                   "band": {k: v["want"] for k, v in rc.items()}, "pass": all(v["ok"] for v in rc.values())})
    return checks


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def timeline_numbers(timeline: Sequence[Mapping[str, Any]], *, shots: int, circuit_ns: float,
                     reset_wait_ns: float) -> dict[str, Any]:
    """Spans of the server's status timeline, and the per-shot cost the execution span implies."""
    first: dict[str, dt.datetime] = {}
    unparsed = 0
    for e in timeline:
        try:
            first.setdefault(str(e["status"]), _ts(e["timestamp"]))
        except (TypeError, KeyError, ValueError):
            unparsed += 1
    out: dict[str, Any] = {"statuses": list(first), "unparsed_entries": unparsed}
    if first:
        out["total_s"] = (max(first.values()) - min(first.values())).total_seconds()
    for name, a, b in (("queue_s", "received", "execution_started"), ("compile_s", "compilation_started", "compilation_ended"),
                       ("execution_s", "execution_started", "execution_ended"),
                       ("post_s", "execution_ended", "completed")):
        if a in first and b in first:
            out[name] = (first[b] - first[a]).total_seconds()
    planned = (circuit_ns + reset_wait_ns) * 1e-9
    out["planned_per_shot_s"] = planned
    out["planned_execution_s"] = planned * shots
    if "execution_s" in out:
        per_shot = out["execution_s"] / shots
        out["per_shot_s"] = per_shot
        out["per_shot_beyond_plan_s"] = per_shot - planned
        out["per_shot_beyond_reset_wait_s"] = per_shot - reset_wait_ns * 1e-9
        out["ratio_to_plan"] = per_shot / planned
    return out


def find_job_dir(raw_root: Path = RAW_ROOT) -> Path:
    dirs = [p.parent for p in Path(raw_root).glob(f"*-{TASK}/*/meta.json")]
    if len(dirs) != 1:
        raise SystemExit(f"expected one job under {raw_root}/*-{TASK}/, found {dirs}")
    return dirs[0]


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def _nan_to_none(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _nan_to_none(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_nan_to_none(x) for x in v]
    return v


def tag_json(path: Path, **keys: Any) -> None:
    """Add ``keys`` at the top of a JSON object file (NaN written as null): ``scripts/emerald_dryrun.py``'s."""
    body = json.loads(path.read_text())
    path.write_text(json.dumps(_nan_to_none({**keys, **body}), indent=1) + "\n")


def per_round(dets: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """The detection fraction at each detector round t."""
    return {str(k): float(dets[:, t == k].mean()) for k in sorted(set(t.tolist()))}


def detector_rounds(ds: Path) -> np.ndarray:
    with open(ds / "detectors.csv") as f:
        return np.array([int(row["round"]) for row in csv.DictReader(f)])


def model_prediction(spec: Any, dec: Any, t: np.ndarray, *, shots: int = MODEL_SHOTS, seed: int = MODEL_SEED) -> dict[str, Any]:
    """The decoder's own noise model, sampled: what it predicts for the pilot (seeded, reproducible)."""
    raw = dec.compile_sampler(seed=seed).sample(shots).astype(np.uint8)
    dets, obs = D.detection_events(spec.stim, raw)
    pred = D.decode(dec, dets)
    k = int((pred[:, 0] != obs[:, 0]).sum())
    k_raw = int(obs[:, 0].sum())
    lo, hi = stats.wilson(k, shots)
    return {"shots": shots, "seed": seed, "detection_fraction": float(dets.mean()),
            "per_round_detection_rate": per_round(dets, t),
            "p_l": k / shots, "p_l_wilson95": [float(lo), float(hi)], "p_always_0": k_raw / shots,
            "all_zero_rows": int((~raw.any(axis=1)).sum())}


def garnet_reference() -> dict[str, Any] | None:
    """Garnet's pilot (InfoRecord/derived/E1/p0_E1_d3_Z_r03_pA/pilot_checks.json): its ratio and per-shot span."""
    if not GARNET_PILOT_CHECKS.is_file():
        return None
    g = json.loads(GARNET_PILOT_CHECKS.read_text())
    f = next(c["value"] for c in g["stop_go"] if c["check"] == "detection fraction")
    tn = g.get("timeline_numbers") or {}
    return {"source": _rel(GARNET_PILOT_CHECKS, REPO), "job_id": g["job_id"], "detection_fraction": f,
            "with_idle_prediction": g["detection_fraction_reference"],
            "ratio": f / g["detection_fraction_reference"], "per_shot_s": tn.get("per_shot_s"),
            "planned_per_shot_s": tn.get("planned_per_shot_s")}


def analyze(job_dir: Path, before: Path, after: Path, *, root: Path, ledger: Path | None, simulated: bool = False,
            p: Pilot | None = None) -> dict[str, Any]:
    p = p or Pilot()
    spec, suite = p.spec, p.suite
    verify = offline_checks(p)
    raw, meta = D.load_memory(job_dir, 0, circuit=spec.stim)
    plan_pilot = json.loads(JOB_PLAN.read_text())["pilot"]
    assert plan_pilot["circuit"] == CIRCUIT and plan_pilot["device"] == DEVICE, plan_pilot["circuit"]
    band = pilot_band(plan_pilot)
    dets, obs = D.detection_events(spec.stim, raw)
    checks = stop_go(raw, dets, band, shots=SHOTS)
    snaps = {"before": load_emerald_snapshot(before), "after": load_emerald_snapshot(after)}
    cal = EM.live_calibration(before)
    if cal.calibration_set_id != meta["calibration_set_id"] and not (simulated and not meta["calibration_set_id"]):
        raise SystemExit(f"the before snapshot is {cal.calibration_set_id}, the job ran on {meta['calibration_set_id']}")
    want_map = [{str(k): v for k, v in sorted(m.items())} for m in p.qubit_mapping]
    rows = None if simulated else [r for r in hw.read_ledger(ledger) if r.job_id == meta["job_id"]]
    checks += qubit_checks(meta, rows, calset=meta["calibration_set_id"], want_mapping=want_map)

    durations = dev.Durations.from_calibration(cal)
    idle = EM.readout_window_ns(spec, durations)
    nm = EM.calibrated_noise(cal, spec, idle_ns=idle)
    dec = spec.noisy(nm)
    assert dec.num_detectors == spec.stim.num_detectors and dec.num_measurements == spec.stim.num_measurements
    dem = dec.detector_error_model(decompose_errors=True)
    distance = len(dec.shortest_graphlike_error())
    note = (f"dqiqm.experiments.emerald.calibrated_noise(the before snapshot, {_rel(before, REPO)}, calibration set "
            f"{cal.calibration_set_id}, spec, idle_ns={idle:g}): device.calibrated_noise(idle_ns=0) (gate depolarizing at "
            "the RB/IRB figures plus relaxation over each gate, the mean readout assignment error as record flips) plus "
            f"the T1/T2 twirl over the {idle:g} ns readout window counted once (data p_xy + p_z, ancillas p_xy)")

    labels = SU.split(JOB_LABEL, 0, len(raw))
    entry = p.entry
    files = {"stim": f"circuits/E1/{CIRCUIT}.stim", "qasm": f"circuits/E1/{CIRCUIT}.qasm",
             "qasm_sha256": entry["qasm_sha256"], "native_sha256": entry["native_sha256"]}
    ds, numbers = IR.write_dataset(
        root, spec, raw, run_id=RUN_ID, batch=0, job_id=meta["job_id"], position_in_job=0, arch=suite.arch, cal=cal,
        durations=durations, circuit_files=files,
        raw_ref={"path": _rel(job_dir / "memory.npz", root), "key": "c0", "meta": _rel(job_dir / "meta.json", root)},
        calibration_paths={"before": _rel(before, root), "after": _rel(after, root)},
        decoder_circuit=dec, decoder_note=note, split_labels=labels, split_info=SU.split_rule(JOB_LABEL, 0, labels),
        simulated=simulated, extra={"device": DEVICE, "job_timeline": meta.get("timeline")},
    )
    label = "SIMULATED" if simulated else "HARDWARE"
    der, summary = IR.write_derived(root, spec, RUN_ID, ds)
    tag_json(der / "summary.json", label=label, device=DEVICE)
    point = {"circuit": CIRCUIT, "dataset": _rel(ds, root), "rounds": spec.rounds, "shots": numbers["shots"],
             "failures": numbers["pm_failures"], "raw_failures": numbers["obs_flips"]}
    ler_dir, _ = IR.write_ler(root, "E1", RUN_ID, [point], title=f"{label}: Emerald E1 d=3 Z basis, patch eA, pilot EP0",
                              label=RUN_ID, fit=False)
    tag_json(ler_dir / "ler_fits.json", label=label, device=DEVICE, decoder="pymatching on the noisy copy of the circuit that ran",
             noise=note, zero_failure_points=[spec.rounds] if numbers["pm_failures"] == 0 else [])
    problems = IR.check_run(ds, der)

    n, k, k_raw, r = numbers["shots"], numbers["pm_failures"], numbers["obs_flips"], spec.rounds
    lo, hi = stats.wilson(k, n)
    rlo, rhi = stats.wilson(k_raw, n)
    e, elo, ehi = (float(x[0]) for x in stats.eps_single(np.array([k]), np.array([n]), np.array([r])))
    t = detector_rounds(ds)
    model = model_prediction(spec, dec, t)
    f = float(dets.mean())
    reference = plan_pilot["calibrated_stim_with_idle"]["detection_fraction"]
    reset_wait = float(cal.reset_wait_ns or 0.0)
    circuit_ns = C.schedule(spec.hardware, durations).duration_ns
    timeline = meta.get("timeline") or []
    result = {
        "format": IR.FORMAT,
        "label": label,
        "task": TASK,
        "job_id": meta["job_id"],
        "device": meta["device"],
        "calibration_set_id": meta["calibration_set_id"],
        "circuit": CIRCUIT,
        "run_id": RUN_ID,
        "raw": _rel(job_dir, root),
        "snapshots": {"before": _rel(before, REPO), "after": _rel(after, REPO),
                      **({k_: {"calibration_set_id": s["calibration_set_id"],
                               "fetched_utc_end": s["fetched_utc_end"]} for k_, s in snaps.items()})},
        "frozen": {"rebuilt_from": _rel(EM.DEFAULT_SNAPSHOT, REPO), "calibration_set_id": p.cal.calibration_set_id,
                   "native_sha256_equal": verify["native_sha256"] == entry["native_sha256"]},
        "verify": verify,
        "stop_go": checks,
        "stop_go_pass": all(c["pass"] for c in checks),
        "detection_fraction_reference": reference,
        "detection_fraction_reference_source": "job_plan.json: pilot.calibrated_stim_with_idle (calibration set "
                                               f"{json.loads(JOB_PLAN.read_text())['calibration_set_id']}, 100,000 shots, "
                                               "seed 2026)",
        "detection_fraction_ratio": f / reference,
        "garnet_reference": garnet_reference(),
        "timeline": timeline,
        "timeline_numbers": timeline_numbers(timeline, shots=n, circuit_ns=circuit_ns, reset_wait_ns=reset_wait)
        if timeline else None,
        "plan_per_shot": {"circuit_ns": circuit_ns, "reset_wait_ns": reset_wait},
        "decoder": {"noise": note, "idle_ns": idle, "dem_errors": dem.num_errors, "graphlike_distance": distance},
        "decode": {"rounds": r, "shots": n, "pm_failures": k, "p_l": k / n, "p_l_wilson95": [float(lo), float(hi)],
                   "raw_failures": k_raw, "p_always_0": k_raw / n, "p_always_0_wilson95": [float(rlo), float(rhi)],
                   "eps_single_point": e, "eps_single_point_95": [elo, ehi],
                   "eps_note": "eps = (1 - (1 - 2 P_L)^(1/R)) / 2 from R = 3 alone: one point, not a fit (the fit needs "
                               "the batches' round counts)"},
        "per_round_detection_rate": per_round(dets, t),
        "model": model,
        "record_problems": problems,
        "paths": {"dataset": _rel(ds, root), "derived": _rel(der, root), "ler": _rel(ler_dir, root)},
        "created_utc": utc_now(),
        "commit": hw.git_head(),
    }
    (der / "pilot_checks.json").write_text(json.dumps(_nan_to_none(result), indent=1, default=str) + "\n")
    return result


def cmd_analyze(args: argparse.Namespace) -> int:
    job_dir = Path(args.raw) if args.raw else find_job_dir()
    res = analyze(job_dir, Path(args.before), Path(args.after), root=Path(args.root), ledger=Path(args.ledger),
                  simulated=args.simulated)
    log(f"{res['label']} job {res['job_id']} on {res['device']}, calibration set {res['calibration_set_id']}")
    for c in res["stop_go"]:
        # long lists: say they matched
        planned = ("qubit" in c["check"] or "ledger" in c["check"]) and c["pass"] and c["value"] is not None
        log(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}: "
            + ("as planned" if planned else f"{c['value']}  band {c['band']}"))
    log(f"STOP/GO: {'PASS' if res['stop_go_pass'] else 'FAIL'}")
    g = res["garnet_reference"]
    log(f"detection fraction / with-idle prediction: {res['detection_fraction_ratio']:.3f}"
        + (f" (Garnet: {g['ratio']:.3f})" if g else ""))
    d, m = res["decode"], res["model"]
    log(f"P_L(R=3) PyMatching {d['pm_failures']}/{d['shots']} = {d['p_l']:.4f} [{d['p_l_wilson95'][0]:.4f}, "
        f"{d['p_l_wilson95'][1]:.4f}]; always-0 {d['raw_failures']}/{d['shots']} = {d['p_always_0']:.4f} "
        f"[{d['p_always_0_wilson95'][0]:.4f}, {d['p_always_0_wilson95'][1]:.4f}]")
    log(f"model ({m['shots']} shots, seed {m['seed']}): P_L {m['p_l']:.4f} [{m['p_l_wilson95'][0]:.4f}, "
        f"{m['p_l_wilson95'][1]:.4f}], always-0 {m['p_always_0']:.4f}, detection fraction {m['detection_fraction']:.5f}")
    log(f"eps single point (R=3): {d['eps_single_point']:.5f} [{d['eps_single_point_95'][0]:.5f}, {d['eps_single_point_95'][1]:.5f}]")
    log("per round, hardware: " + json.dumps(res["per_round_detection_rate"]))
    log("per round, model:    " + json.dumps(m["per_round_detection_rate"]))
    log(f"timeline: {json.dumps(res['timeline_numbers'])}")
    log(f"record problems: {res['record_problems'] or 'none'}; wrote {res['paths']}")
    return 0 if res["stop_go_pass"] and not res["record_problems"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot", help="live, read-only: resonance_check.py, then the patch check")
    v = sub.add_parser("verify", help="rebuild the frozen EP0 and check it; --live also builds the run request read-only")
    v.add_argument("--snapshot", type=Path, help="a live Emerald snapshot (data/calibration/emerald/...)")
    v.add_argument("--live", action="store_true")
    s = sub.add_parser("submit", help="the one job (armed by DQ_QPU=run ...)")
    s.add_argument("--snapshot", type=Path, required=True, help="the before snapshot (its calibration set is used)")
    s.add_argument("--timeout", type=float, default=None, help="seconds to wait for the result (default: iqm-client's 3 h)")
    r = sub.add_parser("recover", help="read-only: re-pull a submitted job and save it as save_raw does")
    r.add_argument("--job-id", required=True)
    r.add_argument("--timeout", type=float, default=None)
    a = sub.add_parser("analyze", help="offline: stop/go, timeline, decode, InfoRecord")
    a.add_argument("--raw", type=Path, help="the job directory (default: the one under InfoRecord/emerald/raw/*-emerald-pilot/)")
    a.add_argument("--before", type=Path, required=True)
    a.add_argument("--after", type=Path, required=True)
    a.add_argument("--root", type=Path, default=RECORD, help="the record root (default InfoRecord/emerald/)")
    a.add_argument("--ledger", type=Path, default=Path(hw.DEFAULT_LEDGER), help="the QPU ledger (MINOR 3's qubit check)")
    a.add_argument("--simulated", action="store_true", help="a rehearsal on simulated raw data (label SIMULATED)")
    args = ap.parse_args()
    return {"snapshot": cmd_snapshot, "verify": cmd_verify, "submit": cmd_submit, "recover": cmd_recover,
            "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
