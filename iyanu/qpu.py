"""Guarded path to IQM Resonance for the iyanu experiments: read freely, submit only when armed.

Reading (token only): `backend(device)` returns a backend whose `run` refuses, with Qiskit's
default scheduling stage (see iyanu.garnet_sim.fake_garnet for why), plus `calibration(...)`.

Submitting: `submit(...)` refuses, before any network call, unless DQ_QPU=run and
DQ_QPU_MAX_SHOTS is set; shots are total (circuits x shots) summed over the ledger. It builds
the run request once, checks every CZ in that request against the intended physical pairs
and the device couplers, submits that same request, appends the ledger line as soon as the
job id returns, then saves the raw result under iyanu/hw_data/<YYYYMMDD>-<task>/<job id>/.
The token is IQM_TOKEN from the environment or .env, and is never printed.
"""
from __future__ import annotations
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

SERVER_URL = "https://resonance.iqm.tech"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT/"iyanu"/"hw_data"
LEDGER = DATA/"ledger.jsonl"


class QPUGuardError(RuntimeError):
    pass


def _token():
    token = os.environ.get("IQM_TOKEN")
    if not token and (ROOT/".env").exists():
        for line in (ROOT/".env").read_text().splitlines():
            if line.startswith("IQM_TOKEN="):
                token = line.split("=", 1)[1].strip()
    if not token:
        raise QPUGuardError("IQM_TOKEN is not set (environment or .env).")
    return token


def backend(device):
    """Read-only Resonance backend: transpile and inspect with it; its run() refuses."""
    from iqm.qiskit_iqm import IQMProvider
    # iqm-qubit-selector's CalibrationDataManager builds its own client that reads only the
    # IQM_TOKEN environment variable, and iqm-client refuses a token given both ways, so the
    # token goes into this process's environment and nowhere else.
    os.environ["IQM_TOKEN"] = _token()
    b = IQMProvider(SERVER_URL, quantum_computer=device).get_backend()

    class ReadOnly(type(b)):
        def get_scheduling_stage_plugin(self):
            return "default"

        def run(self, *_a, **_k):
            raise QPUGuardError("Read-only backend: submit with iyanu.qpu.submit.")

    b.__class__ = ReadOnly
    return b


def calset(b):
    """Calibration set the backend transpiles against and submits with."""
    return str(getattr(b, "_calibration_set_id", None) or b.architecture.calibration_set_id)


def couplings(b):
    return sorted({tuple(sorted((b.index_to_qubit_name(x), b.index_to_qubit_name(y))))
                   for x, y in b.coupling_map.get_edges()})


def calibration(b, *, measure_ns):
    """Live calibration as a CalibrationSnapshot (organisers' fetch, IQM CalibrationDataManager)."""
    from challenge.iqm_utils.get_calibration_data import fetch_calibration
    durations = {"1q": 20., "2q": 40., "measure": measure_ns, "reset": measure_ns}
    snap = fetch_calibration(b, durations_ns=durations, coherence_unit="us", reset_error=0.)
    snap.metadata.update(device=b.name, calibration_set_id=calset(b),
                         assumed_durations_ns=durations)
    return snap


def expected_cz_pairs(prepared):
    """Multiset of physical CZ pairs the placed Stim circuit implies."""
    from challenge.build_emerald_qubit_rotated import interaction_pairs
    mapping = prepared["placement"].stim_to_physical
    return Counter(frozenset((mapping[a], mapping[b])) for a, b in interaction_pairs(prepared["stim_circuit"]))


def check_request(run_request, prepared_list, b):
    """Every CZ in the serialized request is on a coupler and matches the placement exactly."""
    edges = {frozenset(e) for e in couplings(b)}
    report = []
    for circ, prepared in zip(run_request.circuits, prepared_list):
        sent, qubits = Counter(), set()
        for inst in circ.instructions:
            qubits.update(inst.locus)
            if inst.name == "cz":
                pair = frozenset(inst.locus)
                if pair not in edges:
                    raise QPUGuardError(f"{circ.name}: CZ on {sorted(pair)} is not a device coupler.")
                sent[pair] += 1
            elif inst.name not in {"prx", "measure", "barrier"}:
                raise QPUGuardError(f"{circ.name}: non-native instruction {inst.name}.")
        if sent != expected_cz_pairs(prepared):
            raise QPUGuardError(f"{circ.name}: CZ pairs in the request differ from the placement.")
        report.append(dict(name=circ.name, cz=sum(sent.values()), qubits=sorted(qubits)))
    return report


def ledger_usage(task, path=LEDGER):
    if not path.exists():
        return 0, 0
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["task"] == task]
    return len(rows), sum(r["circuits"]*r["shots"] for r in rows)


def _armed(task, circuits, shots, env=None):
    env = os.environ if env is None else env
    if env.get("DQ_QPU") != "run":
        raise QPUGuardError("Not armed: set DQ_QPU=run to submit.")
    cap = int(env.get("DQ_QPU_MAX_SHOTS", "0") or 0)
    _, used = ledger_usage(task)
    if used+circuits*shots > cap:
        raise QPUGuardError(f"Task {task}: {used} shots used + {circuits*shots} requested > cap {cap}.")


def compilation_options(*, dd=False, heralding=False, active_reset=None):
    """IQM server-side options: dynamical decoupling, zeros heralding, active reset before each shot."""
    from iqm.iqm_client import CircuitCompilationOptions, DDMode, HeraldingMode
    opts = CircuitCompilationOptions(dd_mode=DDMode.ENABLED if dd else DDMode.DISABLED,
                                     heralding_mode=HeraldingMode.ZEROS if heralding else HeraldingMode.NONE,
                                     active_reset_cycles=active_reset)
    return opts, dict(dd=bool(dd), heralding=bool(heralding), active_reset_cycles=active_reset)


def submit(b, prepared_list, *, shots, task, purpose, calibration_snapshot=None, options=None):
    """Submit one job (one circuit per prepared entry), wait, save raw data; returns its directory."""
    from iqm.qiskit_iqm.iqm_job import IQMJob
    from challenge.internal_helpers import counts_to_measurement_array
    circuits = [p["circuit"] for p in prepared_list]
    _armed(task, len(circuits), shots)
    opts, opts_meta = options if options is not None else compilation_options()
    request = b.create_run_request(circuits, shots=shots, circuit_compilation_options=opts)
    sent = check_request(request, prepared_list, b)
    circuit_job = b.client.submit_run_request(request)
    job_id = str(circuit_job.job_id)
    now = datetime.now(timezone.utc)
    physical = sorted({q for s in sent for q in s["qubits"]}, key=lambda q: int(q[2:]))
    DATA.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:  # before waiting: the job exists now
        f.write(json.dumps(dict(utc=now.isoformat(), task=task, job_id=job_id, device=b.name,
                                calibration_set_id=calset(b), circuits=len(circuits),
                                shots=shots, physical_qubits=physical, purpose=purpose, options=opts_meta))+"\n")
    print(f"submitted job {job_id}: {len(circuits)} circuit(s) x {shots} shots on {b.name}", flush=True)

    job = IQMJob(b, circuit_job)
    out = DATA/f"{now:%Y%m%d}-{task}"/job_id
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(job_id=job_id, task=task, device=b.name, calibration_set_id=calset(b),
                shots=shots, circuits=len(circuits), purpose=purpose, commit=_git_head(), options=opts_meta,
                submitted_utc=now.isoformat(), sent=sent,
                rounds=[p["rounds"] for p in prepared_list],
                bases=[p.get("basis", "Z") for p in prepared_list],
                stim_to_physical=[{str(k): v for k, v in p["placement"].stim_to_physical.items()} for p in prepared_list])
    if calibration_snapshot is not None:
        calibration_snapshot.save(out/"calibration.json")
    for p, s in zip(prepared_list, sent):
        (out/f"{s['name']}.stim").write_text(str(p["stim_circuit"]))
    try:
        result = job.result()
    except Exception as exc:
        meta.update(success=False, error=repr(exc), saved_utc=datetime.now(timezone.utc).isoformat())
        (out/"meta.json").write_text(json.dumps(meta, indent=2))
        raise
    counts = [result.get_counts(i) for i in range(len(circuits))]
    # Heralding drops shots whose pre-circuit measurement is not all zeros: keep what came back.
    kept = [int(sum(c.values())) for c in counts]
    arrays = {f"c{i}": counts_to_measurement_array(c, p["stim_circuit"].num_measurements, k).astype(np.uint8)
              for i, (c, p, k) in enumerate(zip(counts, prepared_list, kept))}
    meta["shots_returned"] = kept
    np.savez_compressed(out/"measurements.npz", **arrays)
    (out/"counts.json").write_text(json.dumps(counts))
    meta.update(success=True, saved_utc=datetime.now(timezone.utc).isoformat(),
                note="measurements.npz rows are rebuilt from counts (Stim measurement order); shot order is not preserved")
    (out/"meta.json").write_text(json.dumps(meta, indent=2))
    return out


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None
