"""garnet-pilot: the first hardware job. P0, ``E1_d3_Z_r03_pA``, 1 circuit x 1,000 shots on IQM Garnet.

    uv run python scripts/garnet_pilot.py snapshot                        # live, read-only: resonance_check.py + patch check
    uv run python scripts/garnet_pilot.py verify [--snapshot S] [--live]  # offline; --live also builds the run request
    DQ_QPU=run DQ_QPU_TASK=garnet-pilot DQ_QPU_MAX_JOBS=1 DQ_QPU_MAX_SHOTS=1000 \\
        uv run python scripts/garnet_pilot.py submit --snapshot S         # the one job
    uv run python scripts/garnet_pilot.py recover --job-id J              # read-only; only if submit failed after the job id
    uv run python scripts/garnet_pilot.py analyze --before S --after T    # offline: stop/go, timeline, decode, InfoRecord

``snapshot`` runs ``scripts/resonance_check.py`` live (read-only), which writes new snapshots of
Garnet and Emerald under ``data/calibration/``, then checks that the live Garnet calibration still
serves ``prx`` and ``measure`` on the patch's 17 qubits and ``cz`` on its 24 couplers.

``verify`` rebuilds the suite from the committed a18a947f snapshot
(``suite.build(device.live_calibration(device.DEFAULT_SNAPSHOT))``) and checks P0's hardware circuit:
its ``native_sha256`` against ``InfoRecord/circuits/E1/placement.json``, its ops, its couplers, its
33 measurements (measurement j in clbit j on its placed qubit), no SWAP, and the qubits it will be
sent to. How ``hw.submit`` fixes the qubit names (iqm-client 35.0.3, read from the code):

* ``hw.submit`` calls ``IQMBackend.create_run_request(circuits, shots, qubit_index_to_name=...)``,
  which only serializes: no transpile, no layout pass. ``serialize_circuit`` hands the circuit to
  ``qiskit_to_iqm.serialize_instructions``. P0's transpiled circuit has a layout whose registers
  are its own (``q`` 17 and ``ancilla`` 3), so each instruction's qubit is mapped through the
  layout to its physical index, then through ``qubit_index_to_name`` to a device name. We pass the
  map the circuit was placed and checked with (``HardwareCircuit.qubit_index_to_name``), so the live
  backend's own index order never enters.
* ``IQMClient.create_run_request`` then validates the named loci against the calibration's dynamic
  architecture and builds a ``RunRequest`` with ``qubit_mapping=None``: the loci are physical names.
  The request carries no layout or routing option; its options are heralding NONE, DD DISABLED,
  ``active_reset_cycles=None`` (reset by waiting) and the MOVE flags, which Garnet ignores.

``verify --live`` (and ``submit``) build that request with the read-only ``hw.backend(...)``, whose
``run`` and ``submit_*`` refuse, and check it: the serialized instructions hash to the planned
``native_sha256``, ``hw._sent_qubits`` equals the 17 placement qubits, ``qubit_mapping`` is None.

``submit`` refuses unless it is armed exactly for this task (1 job, 1,000 shots, the default
ledger), no ``garnet-pilot`` row is in the ledger, ``InfoRecord/raw/*-garnet-pilot/`` holds no job,
and the server's default calibration set is still the snapshot's. Then it calls ``hw.submit`` once,
with the snapshot's calibration set id, and prints the job id the moment ``hw.submit`` writes the
ledger row (a read-only watch on the ledger), before the result. If anything fails after that,
``recover`` re-pulls the job read-only with ``hw.retrieve`` and saves it with ``hw.save_raw`` into
the same layout. It never submits.

``analyze`` reads the raw job, runs the stop/go checks against ``job_plan.json``'s ``pilot.band``,
reads the job timeline from ``meta.json``, decodes with PyMatching on the noisy copy of the circuit
(priors ``calibrated_noise(<before snapshot>, idle_ns=<readout window>)``), and writes
``InfoRecord/datasets/E1/p0_E1_d3_Z_r03_pA/``, ``derived/`` (plus ``pilot_checks.json``) and ``ler/``,
labelled HARDWARE. Nothing here needs the sim lock: the suite builds in about 17 s, and decoding
1,000 shots is instant.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
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
from dqiqm.experiments import suite as SU

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "InfoRecord"
RAW_ROOT = RECORD / "raw"
CALIBRATION = REPO / "data" / "calibration"
PLACEMENT = RECORD / "circuits" / "E1" / "placement.json"
JOB_PLAN = RECORD / "circuits" / "job_plan.json"

TASK = "garnet-pilot"
DEVICE = "garnet"
CIRCUIT = "E1_d3_Z_r03_pA"
SHOTS = 1000
JOB_LABEL = "P0"
RUN_ID = "p0_" + CIRCUIT
PURPOSE = f"garnet-pilot P0 {CIRCUIT}"
DESIGN_CALSET = "a18a947f-0408-4628-81ea-1715637e434f"
#: Garnet's single zero-SWAP qubit set for d = 3 (the brief): QB1, 2, 4-16, 19, 20.
PATCH_QUBITS = tuple(f"QB{i}" for i in (1, 2, *range(4, 17), 19, 20))
NATIVE_OPS = {"prx", "cz", "measure", "barrier"}
#: Exactly the arming the brief grants, and nothing wider.
ARMING = {"DQ_QPU": "run", "DQ_QPU_TASK": TASK, "DQ_QPU_MAX_JOBS": "1", "DQ_QPU_MAX_SHOTS": str(SHOTS)}


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
    """The 17 device qubits and the 24 couplers (one per ancilla-data CX) of P0's placement."""
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


def cmd_snapshot(args: argparse.Namespace) -> int:
    before = _snapshot_files()
    log(f"{utc_now()} running scripts/resonance_check.py live (read-only)")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "resonance_check.py")], cwd=REPO)
    new = sorted(_snapshot_files() - before)
    for p in new:
        log(f"new snapshot: {p.relative_to(REPO)}")
    garnet = [p for p in new if p.parent.name == "garnet"]
    if proc.returncode != 0 or len(garnet) != 1:
        log(f"STOP: resonance_check.py exited {proc.returncode}, {len(garnet)} new garnet snapshot(s)")
        return 2
    snap = json.loads(garnet[0].read_text())
    calset = snap["calibration_set_id"]
    log(f"SNAPSHOT {garnet[0].relative_to(REPO)}")
    log(f"  calibration_set_id {calset}; server default at fetch {snap.get('default_calibration_set_id_at_fetch')}")
    log(f"  fetched {snap['fetched_utc_start']} .. {snap['fetched_utc_end']}")
    log(f"  same as the design's {DESIGN_CALSET}: {calset == DESIGN_CALSET}")
    qubits, couplers = patch_from_placement(placement_entry())
    missing = patch_gates(snap["architecture"], qubits, couplers)
    log(f"  patch: {len(qubits)} qubits, {len(couplers)} couplers; missing {missing}")
    if any(missing.values()):
        log("STOP: the live calibration lacks a gate the patch needs")
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
    """P0 rebuilt from the committed snapshot, and everything ``submit`` and ``analyze`` need from it."""

    def __init__(self) -> None:
        self.cal = dev.live_calibration(dev.DEFAULT_SNAPSHOT)
        self.suite = SU.build(self.cal)
        self.spec = self.suite[CIRCUIT]
        self.hc = self.spec.hardware
        self.entry = placement_entry()
        self.qubit_index_to_name = dict(self.hc.qubit_index_to_name)
        self.qubits, self.couplers = patch_from_placement(self.entry)
        # what hw.submit will record in meta.json (qubit_mapping) and the ledger (physical qubits)
        self.qubit_mapping = hw._qubit_mappings(None, [self.hc.circuit], self.qubit_index_to_name)  # noqa: SLF001
        self.sent = [sorted({q for i in C.sent(self.hc) for q in i.locus}, key=dev.qubit_sort_key)]


def offline_checks(p: Pilot) -> dict[str, Any]:
    """Raise ``AssertionError`` on the first failed check; return the numbers."""
    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey

    hc, spec, entry = p.hc, p.spec, p.entry
    assert p.cal.calibration_set_id == DESIGN_CALSET, f"the committed snapshot is {p.cal.calibration_set_id}"
    assert hc.target == "hardware", hc.target
    native = C.native_sha256(hc)
    assert native == entry["native_sha256"], f"native_sha256 {native} != placement.json {entry['native_sha256']}"
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
    arch_couplers = {frozenset(c) for c in p.suite.arch.couplers}
    cz = [frozenset(i.locus) for i in sent if i.name == "cz"]
    assert all(c in arch_couplers for c in cz), "a cz off the device's couplers"
    assert {frozenset(c) for c in p.couplers} == set(cz), "the cz loci are not the patch's 24 couplers"
    assert len(set(cz)) == 24 and len(cz) == 72, (len(set(cz)), len(cz))
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
        "measurements": len(meas),
        "swaps": 0,
        "qubits": qubits,
        "qubit_index_to_name": {str(k): v for k, v in sorted(p.qubit_index_to_name.items())},
        "qubit_mapping": {str(k): v for k, v in sorted(mapping.items())},
    }


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


def load_garnet_snapshot(path: Path) -> dict[str, Any]:
    snap = json.loads(Path(path).read_text())
    if snap.get("device") != DEVICE:
        raise SystemExit(f"{path}: device {snap.get('device')!r}, not {DEVICE}")
    return snap


def cmd_verify(args: argparse.Namespace) -> int:
    p = Pilot()
    out = {"offline": offline_checks(p)}
    log("OFFLINE OK: " + json.dumps(out["offline"]))
    if args.snapshot:
        snap = load_garnet_snapshot(args.snapshot)
        missing = patch_gates(snap["architecture"], p.qubits, p.couplers)
        assert not any(missing.values()), missing
        log(f"PATCH OK in {args.snapshot}: calset {snap['calibration_set_id']}")
        if args.live:
            out["run_request"] = run_request_checks(live_run_request(p, snap["calibration_set_id"]), p,
                                                    snap["calibration_set_id"])
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
    """Anything already under ``<raw_root>/*-garnet-pilot/``."""
    return sorted(p for d in Path(raw_root).glob(f"*-{TASK}") for p in d.iterdir())


def ledger_rows(ledger: Path, task: str = TASK) -> list[hw.LedgerRow]:
    return [r for r in hw.read_ledger(ledger) if r.task == task] if Path(ledger).is_file() else []


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
    with LedgerWatch(ledger) as watch:
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
            log(f"RECOVER (read-only, never submit again): uv run python scripts/garnet_pilot.py recover --job-id {job_id}")
        return {"job_id": job_id, "error": f"{type(error).__name__}: {error}", "ok": False}
    row_check = check_ledger_row(rows[0], job_id=sub.job_id, calset=calset, qubits=p.sent[0])
    meta = json.loads((sub.path / "meta.json").read_text())
    meta_check = {
        "qubit_mapping": meta["qubit_mapping"] == [{str(k): v for k, v in sorted(m.items())} for m in p.qubit_mapping],
        "physical_qubits": meta["physical_qubits"] == p.sent,
        "calibration_set_id": meta["calibration_set_id"] == calset,
        "shots": meta["shots"] == SHOTS,
        "success": meta["success"],
        "timeline": bool(meta.get("timeline")),
    }
    ok = len(rows) == 1 and all(v["ok"] for v in row_check.values()) and all(meta_check.values())
    log(f"raw: {sub.path}")
    log("ledger row: " + json.dumps(row_check))
    log("meta.json: " + json.dumps(meta_check))
    return {"job_id": sub.job_id, "path": str(sub.path), "calibration_set_id": sub.calibration_set_id,
            "qubits": sub.qubits, "ledger_rows": len(rows), "ledger_check": row_check, "meta_check": meta_check, "ok": ok}


def cmd_submit(args: argparse.Namespace) -> int:
    check_arming(os.environ)
    ledger = hw.ledger_path()
    snap = load_garnet_snapshot(args.snapshot)
    calset = snap["calibration_set_id"]
    p = Pilot()
    log("OFFLINE OK: " + json.dumps(offline_checks(p)))
    missing = patch_gates(snap["architecture"], p.qubits, p.couplers)
    if any(missing.values()):
        raise SystemExit(f"refusing: {args.snapshot} lacks {missing}")
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


def readout_window_ns(cal: dev.Calibration, spec: Any) -> float:
    """A round's readout window: its slowest ancilla's measure duration (the layer waits for it)."""
    return max(float(cal.qubits[spec.physical[q]]["measure_ns"]) for q, r in spec.roles.items() if r == "ancilla")


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def analyze(job_dir: Path, before: Path, after: Path, *, root: Path, simulated: bool = False,
            p: Pilot | None = None) -> dict[str, Any]:
    p = p or Pilot()
    spec, suite = p.spec, p.suite
    verify = offline_checks(p)
    raw, meta = D.load_memory(job_dir, 0, circuit=spec.stim)
    band = json.loads(JOB_PLAN.read_text())["pilot"]["band"]
    dets, obs = D.detection_events(spec.stim, raw)
    checks = stop_go(raw, dets, band, shots=SHOTS)
    if meta.get("physical_qubits") is not None:
        checks.append({"check": "physical qubits (meta.json)", "value": meta["physical_qubits"], "band": p.sent,
                       "pass": meta["physical_qubits"] == p.sent})
    want_map = [{str(k): v for k, v in sorted(m.items())} for m in p.qubit_mapping]
    checks.append({"check": "qubit_mapping (meta.json)", "value": meta["qubit_mapping"], "band": want_map,
                   "pass": meta["qubit_mapping"] == want_map})

    snaps = {"before": load_garnet_snapshot(before), "after": load_garnet_snapshot(after)}
    cal = dev.live_calibration(before)
    if cal.calibration_set_id != meta["calibration_set_id"] and not (simulated and not meta["calibration_set_id"]):
        raise SystemExit(f"the before snapshot is {cal.calibration_set_id}, the job ran on {meta['calibration_set_id']}")
    idle = readout_window_ns(cal, spec)
    nm = dev.calibrated_noise(cal, spec.physical, spec.roles, pairs=spec.cx_pairs, idle_ns=idle, x_ancillas=spec.x_ancillas)
    dec = spec.noisy(nm)
    assert dec.num_detectors == spec.stim.num_detectors and dec.num_measurements == spec.stim.num_measurements
    dem = dec.detector_error_model(decompose_errors=True)
    distance = len(dec.shortest_graphlike_error())
    note = (f"dqiqm.experiments.device.calibrated_noise(the before snapshot, {_rel(before, REPO)}, calibration set "
            f"{cal.calibration_set_id}, idle_ns={idle:g}): gate depolarizing at the RB/IRB figures plus relaxation over "
            f"each gate, the mean readout assignment error as record flips, and the T1/T2 twirl over each {idle:g} ns "
            "readout window")

    durations = dev.Durations.from_calibration(cal)
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
        simulated=simulated, extra={"device": meta["device"], "job_timeline": meta.get("timeline")},
    )
    der, summary = IR.write_derived(root, spec, RUN_ID, ds)
    problems = IR.check_run(ds, der)
    point = {"circuit": CIRCUIT, "dataset": _rel(ds, root), "rounds": spec.rounds, "shots": numbers["shots"],
             "failures": numbers["pm_failures"], "raw_failures": numbers["obs_flips"]}
    label = "SIMULATED" if simulated else "HARDWARE"
    ler_dir, _ = IR.write_ler(root, "E1", RUN_ID, [point], title=f"{label}: E1 d=3 Z basis, patch pA, pilot P0",
                              label=RUN_ID, fit=False)

    n, k, k_raw, r = numbers["shots"], numbers["pm_failures"], numbers["obs_flips"], spec.rounds
    lo, hi = stats.wilson(k, n)
    rlo, rhi = stats.wilson(k_raw, n)
    e, elo, ehi = (float(x[0]) for x in stats.eps_single(np.array([k]), np.array([n]), np.array([r])))
    reset_wait = float(cal.reset_wait_ns or 0.0)
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
        "verify": verify,
        "stop_go": checks,
        "stop_go_pass": all(c["pass"] for c in checks),
        "detection_fraction_reference": json.loads(JOB_PLAN.read_text())["pilot"]["calibrated_stim_with_idle"]["detection_fraction"],
        "timeline": timeline,
        "timeline_numbers": timeline_numbers(timeline, shots=n, circuit_ns=C.schedule(spec.hardware, durations).duration_ns,
                                             reset_wait_ns=reset_wait) if timeline else None,
        "decoder": {"noise": note, "idle_ns": idle, "dem_errors": dem.num_errors, "graphlike_distance": distance},
        "decode": {"rounds": r, "shots": n, "pm_failures": k, "p_l": k / n, "p_l_wilson95": [float(lo), float(hi)],
                   "raw_failures": k_raw, "p_always_0": k_raw / n, "p_always_0_wilson95": [float(rlo), float(rhi)],
                   "eps_single_point": e, "eps_single_point_95": [elo, ehi],
                   "eps_note": "eps = (1 - (1 - 2 P_L)^(1/R)) / 2 from R = 3 alone: one point, not a fit (the fit needs "
                               "the batches' round counts)"},
        "per_round_detection_rate": per_round_rates(dets, ds),
        "record_problems": problems,
        "paths": {"dataset": _rel(ds, root), "derived": _rel(der, root), "ler": _rel(ler_dir, root)},
        "created_utc": utc_now(),
        "commit": hw.git_head(),
    }
    (der / "pilot_checks.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    return result


def per_round_rates(dets: np.ndarray, ds: Path) -> dict[str, float]:
    """The detection fraction at each detector round t (``detectors.csv``'s ``round``)."""
    with open(ds / "detectors.csv") as f:
        t = np.array([int(row["round"]) for row in csv.DictReader(f)])
    return {str(k): float(dets[:, t == k].mean()) for k in sorted(set(t.tolist()))}


def cmd_analyze(args: argparse.Namespace) -> int:
    job_dir = Path(args.raw) if args.raw else find_job_dir()
    res = analyze(job_dir, Path(args.before), Path(args.after), root=Path(args.root), simulated=args.simulated)
    log(f"{res['label']} job {res['job_id']} on {res['device']}, calibration set {res['calibration_set_id']}")
    for c in res["stop_go"]:
        planned = "qubit" in c["check"] and c["pass"]  # a long list: say it matched
        log(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}: "
            + ("as planned" if planned else f"{c['value']}  band {c['band']}"))
    log(f"STOP/GO: {'PASS' if res['stop_go_pass'] else 'FAIL'}")
    d = res["decode"]
    log(f"P_L(R=3) PyMatching {d['pm_failures']}/{d['shots']} = {d['p_l']:.4f} [{d['p_l_wilson95'][0]:.4f}, "
        f"{d['p_l_wilson95'][1]:.4f}]; always-0 {d['raw_failures']}/{d['shots']} = {d['p_always_0']:.4f} "
        f"[{d['p_always_0_wilson95'][0]:.4f}, {d['p_always_0_wilson95'][1]:.4f}]")
    log(f"eps single point (R=3): {d['eps_single_point']:.5f} [{d['eps_single_point_95'][0]:.5f}, {d['eps_single_point_95'][1]:.5f}]")
    log(f"timeline: {json.dumps(res['timeline_numbers'])}")
    log(f"record problems: {res['record_problems'] or 'none'}; wrote {res['paths']}")
    return 0 if res["stop_go_pass"] and not res["record_problems"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot", help="live, read-only: resonance_check.py, then the patch check")
    v = sub.add_parser("verify", help="rebuild P0 and check it; --live also builds the run request read-only")
    v.add_argument("--snapshot", type=Path, help="a live Garnet snapshot (data/calibration/garnet/...)")
    v.add_argument("--live", action="store_true")
    s = sub.add_parser("submit", help="the one job (armed by DQ_QPU=run ...)")
    s.add_argument("--snapshot", type=Path, required=True, help="the before snapshot (its calibration set is used)")
    s.add_argument("--timeout", type=float, default=None, help="seconds to wait for the result (default: iqm-client's 3 h)")
    r = sub.add_parser("recover", help="read-only: re-pull a submitted job and save it as save_raw does")
    r.add_argument("--job-id", required=True)
    r.add_argument("--timeout", type=float, default=None)
    a = sub.add_parser("analyze", help="offline: stop/go, timeline, decode, InfoRecord")
    a.add_argument("--raw", type=Path, help="the job directory (default: the one under InfoRecord/raw/*-garnet-pilot/)")
    a.add_argument("--before", type=Path, required=True)
    a.add_argument("--after", type=Path, required=True)
    a.add_argument("--root", type=Path, default=RECORD, help="the record root (default InfoRecord/)")
    a.add_argument("--simulated", action="store_true", help="a rehearsal on simulated raw data (label SIMULATED)")
    args = ap.parse_args()
    return {"snapshot": cmd_snapshot, "verify": cmd_verify, "submit": cmd_submit, "recover": cmd_recover,
            "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
