"""emerald-fresh: fresh Emerald E1 shots from the cloud API, an independent check of the frozen decoders. 3 jobs, 45,000 shots.

    uv run python scripts/emerald_fresh.py plan                                 # offline: the allowance against the job plan
    uv run python scripts/emerald_fresh.py snapshot                             # live, read-only: one Emerald snapshot
    uv run python scripts/emerald_fresh.py verify [--snapshot P [--live]]      # the freeze, the live checks, the pinned requests
    DQ_QPU=run DQ_QPU_TASK=emerald-fresh DQ_QPU_MAX_JOBS=3 DQ_QPU_MAX_SHOTS=45000 \\
        uv run python scripts/emerald_fresh.py submit --snapshot P --jobs F0    # the pilot
    DQ_QPU=run DQ_QPU_TASK=emerald-fresh DQ_QPU_MAX_JOBS=3 DQ_QPU_MAX_SHOTS=45000 \\
        uv run python scripts/emerald_fresh.py submit --snapshot P --jobs F1 F2 # only after F0 passed its checks
    uv run python scripts/emerald_fresh.py recover --job-id J                  # read-only; only if submit failed after a job id
    uv run python scripts/emerald_fresh.py analyze --snapshots P Q ...         # offline (sim lock): the records and the fits
    uv run python scripts/emerald_fresh.py doc                                 # offline: docs/emerald-fresh.md

What it is for (Sal, 2026-10-25 02:49 UTC: "Run with fresh emerald shots directly from the cloud backend")
    An independent check of the frozen decoders on shots that did not exist when anything was trained,
    tuned or selected. Nothing is chosen on these shots. Their ``split`` is ``fresh``: never finetune,
    validation or test, and never in the ML training or tuning data.

The jobs (the brief's QPU allowance: 3 jobs, 45,000 shots, DD off as in the batch)
    ``F0``: ``E1_d3_Z_r03_eA`` x 1,000, the sanity pilot. ``F1``: the 11 E1 Z circuits on eA x 2,000, and ``F2``:
    the 11 E1 X circuits on eA x 2,000, in the circuit order of the batch's ``E-B3-1-E1`` and ``E-B3-2-E1``
    (``InfoRecord/emerald/circuits/job_plan.json``). F1 and F2 go only if F0 passed every check.

What is reused (``scripts/batch_run.py``, imported, never copied)
    The freeze (``Frozen``, ``freeze_checks``: every circuit rebuilt from the committed 23e12f6b snapshot, its
    native, Stim and QASM hashes equal to ``placement.json``'s), the live checks against a moved calibration set
    (``live_checks``, which runs ``emerald.check_live_couplers``), the pinned run request (``live_run_request``,
    ``run_request_checks``: the very request ``hw.submit`` builds, built on the read-only backend, circuit by
    circuit equal to the frozen ``native_sha256`` on eA's 17 qubits, DD disabled), ``submit_job`` (manifest
    first, ``hw.submit``, the job id from the ledger row), ``finish_job`` (the ledger row, ``meta.json`` and every
    circuit's sanity), ``resume_state``, ``recover_job``, the sanity references and checks, ``timeline_numbers``,
    ``fit_both``, ``read_points`` and ``pool``. ``Frozen.jobs`` is overridden so these three jobs are the
    segment; ``Segment.split`` is overridden to ``fresh``. The snapshot is ``scripts/emerald_dd.py``'s
    Emerald-only ``take_snapshot`` (``resonance_check.py``'s ``main`` would also write a Garnet one).

The decoder (the brief; ``orchestration/reports/emerald-redecode-handoff.md``)
    PyMatching on the noisy copy of the circuit that ran, with ``device.hardware_noise(cal, spec,
    windows=None)``: the 360 ns ``measure_ns`` readout-window fallback, which emerald-redecode chose on the
    B1+B2 validation shots over the compiled 1,036 ns window. Since emerald-compile, ``hardware_noise``'s
    default (``windows="auto"``) reads the compile file, so ``windows=None`` is passed explicitly, and
    ``batch_run.decoder_priors`` is replaced by this module's in this process, so the sanity references' model
    is the same prior. Every record says ``priors:`` :data:`PRIORS`.

The analysis
    P_L(R) with Wilson 95% per basis (F1, F2), and eps_L per basis by the challenge's fit, 2 P_L = 1 - (1 -
    2 eps_L)^R (A = 1, every R), beside the free-A form, over all R and over R <= 3, with bootstrap 95%
    intervals (1,000 draws, seed 2026). The same fits of the batch's own records (B3 test, B1+B2) are put
    beside them, with a per-R two-proportion comparison against B3. F0 is recorded but not pooled into the fits,
    so the fresh pools have the batch's design (2,000 shots per R).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dqiqm import decode as D
from dqiqm import hw
from dqiqm import inforecord as IR
from dqiqm import stats
from dqiqm.experiments import device as dev

REPO = Path(__file__).resolve().parents[1]


def _load(name: str) -> Any:
    """A sibling script as a private module, ``emerald_fresh_<name>`` (registered first: its dataclasses need
    ``sys.modules``). The private name keeps the ``decoder_priors`` override below off the ``batch_run`` that
    ``tests/test_batch_run.py`` loads."""
    key = f"emerald_fresh_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


BR = _load("batch_run")

TASK = "emerald-fresh"
DEVICE = "emerald"
SPLIT_LABEL = "fresh"
#: The record's ``batch`` field: after B1-B3. The split, not the batch number, keeps these shots out of the ML data.
BATCH = 4
PILOT = "E1_d3_Z_r03_eA"
#: F1 and F2 run the batch's B3 eA jobs' circuits, order and shots.
SOURCE = {"F1": ("E-B3-1-E1", "Z"), "F2": ("E-B3-2-E1", "X")}
PATCH = "eA"
PATCH_QUBITS = tuple(f"QB{i}" for i in (13, 14, *range(19, 23), *range(27, 32), *range(36, 40), 44, 45))
PRIORS = "hardware_noise fallback 360 ns (windows=None; chosen on B1+B2 validation by emerald-redecode 47f0c07)"
DECODER = "pymatching on the noisy copy of the circuit that ran"
SEED, BOOTSTRAP = 2026, 1000
#: The pilot's checks beyond the batch's: the detection fraction within [0.5, 2] x the priors' own, and the
#: decoded P_L at most twice the batch's worst P_L of the same circuit (B1-B3).
PILOT_DETECTION_RATIO = (0.5, 2.0)
PILOT_P_L_RATIO_MAX = 2.0
#: The batch's B3 eps_L (A = 1, all R, fallback prior), from the brief; also read from the records.
B3_EPS = {"X": 0.0366, "Z": 0.0476}

RECORD = BR.DEVICES[DEVICE].record
RAW_ROOT = RECORD / "raw"
SUMMARY_DIR = RECORD / "ler" / "E1" / "fresh_summary"
DOC = REPO / "docs" / "emerald-fresh.md"
PROGRESS = Path("/home/sal/duquantumIQM/orchestration/reports/emerald-fresh-progress.md")

log, utc_now, write_json, tag_json, _rel = BR.log, BR.utc_now, BR.write_json, BR.tag_json, BR._rel


@dataclass(frozen=True)
class FreshSegment(BR.Segment):
    """``batch_run.Segment`` whose split is ``fresh``: the manifests and records say so."""

    @property
    def split(self) -> str:
        return SPLIT_LABEL


SEG = FreshSegment(TASK, DEVICE, BATCH, ("F0", "F1", "F2"), 3, 45_000)


def decoder_priors(cal: dev.Calibration, spec: Any) -> tuple[Any, dict[str, Any], str] | None:
    """``batch_run.decoder_priors`` with the 360 ns fallback forced: ``hardware_noise(cal, spec, windows=None)``."""
    if spec.noisy is None or not spec.stim.num_observables:
        return None
    nm, info = dev.hardware_noise(cal, spec, windows=None)
    note = (f"priors: {PRIORS}. dqiqm.experiments.device.hardware_noise(calibration set {cal.calibration_set_id}, spec, "
            f"windows=None): calibrated_noise with the {info['idle_channel']} idle over {info['idle_ns']:g} ns per "
            f"mid-circuit readout layer (readout window {info['readout_window_ns']:g} ns from "
            f"{info['readout_window_source']}, plus a {info['delay_ns']:g} ns delay), T2 {info['t2']}, final readout "
            f"{info['final_readout']}")
    return spec.noisy(nm), info, note


# batch_run.reference() looks decoder_priors up in its own module: the sanity model is then this prior too.
BR.decoder_priors = decoder_priors


# ---------------------------------------------------------------------------
# The jobs and the frozen circuits.
# ---------------------------------------------------------------------------


def fresh_jobs(plan: Mapping[str, Any], built: Mapping[str, Any] | None = None) -> list[Any]:
    """F0, F1, F2 from ``job_plan.json`` (and the rebuilt ``jobs()``). Raises ``SystemExit`` on any disagreement."""
    entries = {j["label"]: j for j in plan["jobs"]}
    out = [BR.PlannedJob("F0", "E1", BATCH, (PILOT,), 1000, 0)]
    for pos, (label, (src, basis)) in enumerate(SOURCE.items(), start=1):
        j = entries.get(src)
        if j is None:
            raise SystemExit(f"{label}: {src} is not in the job plan")
        order = tuple(j["order"])
        problems = []
        if j["experiment"] != "E1" or j["batch"] != 3 or j["circuits"] != 11 or j["shots"] != 2000:
            problems.append(f"{src} is {j['experiment']} batch {j['batch']}, {j['circuits']} x {j['shots']}")
        if len(set(order)) != 11 or not all(n.startswith(f"E1_d3_{basis}_r") and n.endswith(f"_{PATCH}") for n in order):
            problems.append(f"{src}'s order is not the 11 E1 {basis} circuits on {PATCH}: {order}")
        if built is not None:
            b = built.get(src)
            if b is None or tuple(b.circuits) != order or b.shots != j["shots"]:
                problems.append("the rebuilt jobs() disagree with job_plan.json")
        if problems:
            raise SystemExit(f"{label}: " + "; ".join(problems))
        out.append(BR.PlannedJob(label, "E1", BATCH, order, int(j["shots"]), pos))
    if PILOT not in out[1].names:
        raise SystemExit(f"{PILOT} is not in F1")
    jobs, shots = len(out), sum(j.total_shots for j in out)
    if (jobs, shots) != (SEG.jobs, SEG.shots):
        raise SystemExit(f"the plan gives {jobs} jobs and {shots:,} shots; the brief allows {SEG.jobs} and {SEG.shots:,}")
    return out


class Frozen(BR.Frozen):
    """``batch_run.Frozen`` for Emerald, whose segment is these three jobs."""

    def __init__(self) -> None:
        super().__init__(DEVICE)

    def jobs(self, seg: Any = SEG) -> list[Any]:
        jobs = fresh_jobs(self.plan, self.built_jobs)
        for job in jobs:
            for n in job.names:
                if tuple(self.planned_qubits(n)) != PATCH_QUBITS:
                    raise SystemExit(f"{n}: placed on {self.planned_qubits(n)}, not eA's 17")
        return jobs


def take_snapshot() -> Path:
    return _load("emerald_dd").take_snapshot()


# ---------------------------------------------------------------------------
# The pilot's checks.
# ---------------------------------------------------------------------------


def batch_points(circuit: str) -> list[dict[str, Any]]:
    """The batch's decoded points of ``circuit`` (B1-B3), from the batch's own records."""
    return [p for p in BR.read_points(RECORD, "E1") if p["circuit"] == circuit]


def pilot_checks(fz: Frozen, job_dir: Path, ref: Mapping[str, Any], cal: dev.Calibration) -> list[dict[str, Any]]:
    """F0 beyond the batch's sanity: plausible against the priors' own model and against the batch's P_L."""
    spec = fz.suite[PILOT]
    meta = json.loads((Path(job_dir) / "meta.json").read_text())
    raw, _ = D.load_memory(job_dir, 0, circuit=spec.stim)
    dets, obs = D.detection_events(spec.stim, raw)
    noisy, info, _ = decoder_priors(cal, spec)
    lep = D.logical_error_probability(D.decode(noisy, dets), obs)
    checks = []
    statuses = {str(e.get("status")) for e in meta.get("timeline") or [] if isinstance(e, Mapping)}
    checks.append({"check": "timeline: execution_started and execution_ended", "value": sorted(statuses),
                   "band": ["execution_started", "execution_ended"],
                   "pass": {"execution_started", "execution_ended"} <= statuses})
    model = (ref.get("model") or {}).get("detection_fraction")
    f = float(dets.mean())
    lo, hi = (PILOT_DETECTION_RATIO[0] * model, PILOT_DETECTION_RATIO[1] * model) if model else (None, None)
    checks.append({"check": "detection fraction against the priors' own", "value": f, "band": [lo, hi],
                   "pass": model is not None and lo <= f <= hi,
                   "info": {"model": model, "ratio": f / model if model else None, "idle_ns": info["idle_ns"]}})
    pts = batch_points(PILOT)
    ref_p = max(p["failures"] / p["shots"] for p in pts) if pts else None
    checks.append({"check": "decoded P_L against the batch's worst P_L of the same circuit (B1-B3)", "value": lep.p,
                   "band": f"<= {PILOT_P_L_RATIO_MAX:g} x {ref_p:.4f}" if ref_p else "no batch point",
                   "pass": ref_p is not None and lep.p <= PILOT_P_L_RATIO_MAX * ref_p,
                   "info": {"failures": lep.k, "shots": lep.n, "wilson95": [lep.wilson_lo, lep.wilson_hi],
                            "obs_flips": int(np.asarray(obs)[:, 0].sum()),
                            "batch": [{k: p[k] for k in ("batch", "failures", "shots", "run_id")} for p in pts]}})
    return checks


def pilot_passed(ledger: Path) -> bool:
    """F0 is in the ledger once, saved, and passed both the batch's checks and :func:`pilot_checks`."""
    rows = BR.job_rows(ledger, SEG).get("F0", [])
    if len(rows) != 1:
        return False
    mpath = RAW_ROOT / f"{rows[0].time[:10].replace('-', '')}-{TASK}" / "F0.json"
    if not mpath.exists():
        return False
    man = json.loads(mpath.read_text())
    return bool(man.get("ok")) and bool(man.get("pilot_pass"))


# ---------------------------------------------------------------------------
# plan, snapshot, verify, submit, recover
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    jobs = fresh_jobs(BR.plan_json(DEVICE))
    for j in jobs:
        log(f"{j.label}: {len(j.names)} x {j.shots} = {j.total_shots:,} shots: {', '.join(j.names)}")
    log(f"total {len(jobs)} jobs, {sum(j.total_shots for j in jobs):,} shots; arming: "
        + " ".join(f"{k}={v}" for k, v in SEG.arming().items()))
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    path = take_snapshot()
    snap = BR.load_snapshot(DEVICE, path)
    log(f"SNAPSHOT {_rel(path, REPO)}")
    log(f"  calibration_set_id {snap['calibration_set_id']}; server default at fetch {snap.get('default_calibration_set_id_at_fetch')}")
    log(f"  fetched {snap['fetched_utc_start']} .. {snap['fetched_utc_end']}")
    log(f"  same as the design's {BR.DEVICES[DEVICE].design_calset}: {snap['calibration_set_id'] == BR.DEVICES[DEVICE].design_calset}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    if args.live and not args.snapshot:
        raise SystemExit("--live needs --snapshot (the calibration set to build the requests for)")
    fz = Frozen()
    out = BR.preflight(fz, SEG, args.snapshot, live=args.live)
    for name, ref in out["references"].items():
        m = ref["model"] or {}
        log(f"  ref {name}: model detection {m.get('detection_fraction')} (idle {m.get('idle_ns')} ns)")
    if args.out:
        write_json(Path(args.out), out)
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    BR.check_arming(SEG, os.environ)
    ledger = hw.ledger_path()
    wanted = list(args.jobs)
    if wanted != [x for x in SEG.labels if x in wanted]:
        raise SystemExit(f"refusing: jobs go in the order {SEG.labels}, not {wanted}")
    snapshot = Path(args.snapshot)
    calset = BR.load_snapshot(DEVICE, snapshot)["calibration_set_id"]
    fz = Frozen()
    jobs = {j.label: j for j in fz.jobs(SEG)}
    pre = BR.preflight(fz, SEG, snapshot, live=False)
    refs = pre["references"]
    day = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    pre_path = RAW_ROOT / f"{day}-{TASK}" / "preflight.json"
    if not pre_path.exists():
        write_json(pre_path, pre)
    default_now = hw.calibration_set_id(DEVICE)
    if default_now != calset:
        raise SystemExit(f"refusing: the server's default calibration set is now {default_now}, the snapshot's {calset}; "
                         "take a new snapshot")
    cal = BR.live_calibration(DEVICE, snapshot)
    for label in wanted:
        job = jobs[label]
        if label != "F0" and not pilot_passed(ledger):
            log(f"STOP: {label} goes only after F0 passed its checks")
            return 4
        state = BR.resume_state(fz, SEG, job, ledger=ledger, raw_root=RAW_ROOT, refs=refs)
        if state == "raw missing":
            return 5
        if state == "failed":
            log(f"STOP: {label} failed its checks. Report it.")
            return 4
        if state == "done":
            continue
        now = hw.calibration_set_id(DEVICE)
        if now != calset:
            log(f"{utc_now()} the server's default calibration set moved: {calset} -> {now}; a new snapshot and the live checks")
            new = take_snapshot()
            try:
                live = BR.live_checks(fz, list(jobs.values()), new)
            except AssertionError as e:
                log(str(e))
                return 6
            if live["calibration_set_id"] != now:
                raise SystemExit(f"STOP: the new snapshot is {live['calibration_set_id']}, the server default {now}")
            calset, cal = now, BR.live_calibration(DEVICE, new)
            refs = BR.references(fz, list(jobs.values()), cal)
        try:
            rr = BR.run_request_checks(BR.live_run_request(fz, job, calset), fz, job, calset)
        except AssertionError as e:
            log(f"STOP: the run request for {label} failed its check: {e}")
            return 7
        log(f"RUN REQUEST OK (read-only backend, not submitted) {label}: {json.dumps(rr)}")
        res = BR.submit_job(fz, SEG, job, calset=calset, ledger=ledger, raw_root=RAW_ROOT, refs=refs, timeout=args.timeout)
        log(f"{label} " + ("OK" if res["ok"] else "NOT OK") + ": " + json.dumps(res))
        if label == "F0" and res.get("path"):
            checks = pilot_checks(fz, Path(res["path"]), refs[PILOT], cal)
            ok = all(c["pass"] for c in checks)
            man = json.loads(Path(res["manifest"]).read_text())
            man.update(pilot_checks=checks, pilot_pass=ok, pilot_priors=PRIORS)
            write_json(Path(res["manifest"]), man)
            for c in checks:
                log(f"  PILOT {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {c['value']} (band {c['band']})")
            res["ok"] = res["ok"] and ok
        if not res["ok"]:
            log(f"STOP: {label} failed a check; nothing more is submitted. Report it.")
            return 4
    rows = BR.job_rows(ledger, SEG)
    if all(len(rows.get(x, [])) == 1 for x in SEG.labels):
        seg_file = RAW_ROOT / f"{day}-{TASK}" / "segment.json"
        write_json(seg_file, {
            "format": "dqiqm.batches/1", "task": TASK, "device": DEVICE, "batch": BATCH, "split": SPLIT_LABEL,
            "allowance": {"jobs": SEG.jobs, "shots": SEG.shots}, "before": _rel(snapshot, REPO),
            "jobs": [{"label": j.label, "position": j.position, "job_id": rows[j.label][0].job_id,
                      "ledger_time": rows[j.label][0].time, "calibration_set_id": rows[j.label][0].calibration_set_id,
                      "raw": _rel(BR.find_job_dir(RAW_ROOT, TASK, rows[j.label][0].job_id), RECORD),
                      "order": [{"position": i, "circuit": n, "native_sha256": fz.placement[n]["native_sha256"]}
                                for i, n in enumerate(j.names)]} for j in jobs.values()],
            "written_utc": utc_now()})
        n = len(BR.ledger_rows(ledger, TASK))
        shots = sum(r.total_shots for r in BR.ledger_rows(ledger, TASK))
        log(f"SEGMENT SUBMITTED {TASK}: {n} jobs, {shots:,} shots in the ledger (allowance {SEG.jobs}, {SEG.shots:,}); "
            f"{_rel(seg_file, REPO)}")
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    fz = Frozen()
    ledger = hw.ledger_path()
    job, path = BR.recover_job(fz, SEG, args.job_id, ledger=ledger, raw_root=RAW_ROOT, timeout=args.timeout)
    log(f"RECOVERED (hw.retrieve + hw.save_raw, read-only): {path}")
    row = next(r for r in BR.ledger_rows(ledger, TASK) if r.job_id == args.job_id)
    cal = BR.live_calibration(DEVICE, Path(args.snapshot)) if args.snapshot else fz.cal
    refs = BR.references(fz, [job], cal)
    day = row.time[:10].replace("-", "")
    res = BR.finish_job(fz, SEG, job, path, calset=row.calibration_set_id, ledger=ledger, refs=refs,
                        manifest=BR.manifest_path(RAW_ROOT, SEG, job, day))
    log(f"{job.label} " + ("OK" if res["ok"] else "NOT OK") + ": " + json.dumps(res))
    return 0 if res["ok"] else 4


# ---------------------------------------------------------------------------
# analyze: the records, the fits, the comparison
# ---------------------------------------------------------------------------


def run_id(label: str, name: str) -> str:
    return f"fresh_{label}_{name}"


def split_info(label: str, shots: int) -> dict[str, Any]:
    return {"rule": "fresh: shots taken after every decoder was trained, tuned and selected (emerald-fresh, Sal "
                    "2026-10-25 02:49 UTC); never finetune, validation or test, never in the ML training or tuning data; "
                    "nothing is chosen on them",
            "job": label, "role": SPLIT_LABEL, "counts": {SPLIT_LABEL: int(shots)}}


def two_proportions(k1: int, n1: int, k2: int, n2: int) -> dict[str, Any]:
    """p1 - p2 with the pooled two-sided z test (information: no multiple-comparison correction)."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    z = (p1 - p2) / se if se else 0.0
    return {"diff": p1 - p2, "z": z, "p_two_sided": math.erfc(abs(z) / math.sqrt(2))}


def fits(points: Sequence[Mapping[str, Any]], max_rounds: int | None = None) -> dict[str, Any]:
    chosen = [p for p in points if max_rounds is None or p["rounds"] <= max_rounds]
    f = BR.fit_both(*BR.pool(chosen), bootstrap=BOOTSTRAP)
    f.pop("_fits", None)
    return f


def analyze(snapshots: Sequence[Path], *, root: Path, ledger: Path, fz: Frozen | None = None) -> dict[str, Any]:
    """Every fresh run's dataset, derived features and checks, then the LER records, fits and comparison."""
    fz = fz or Frozen()
    jobs = fz.jobs(SEG)
    rows = BR.job_rows(ledger, SEG)
    cals: dict[str, tuple[Path, dev.Calibration]] = {}
    for p in snapshots:
        c = BR.live_calibration(DEVICE, Path(p))
        cals.setdefault(c.calibration_set_id, (Path(p), c))
    snap_rel = {"before": _rel(Path(snapshots[0]), root), "after": _rel(Path(snapshots[-1]), root)}
    runs, job_out = [], []
    for job in jobs:
        jr = rows.get(job.label, [])
        if len(jr) != 1:
            raise SystemExit(f"{job.label}: {len(jr)} ledger rows")
        job_dir = BR.find_job_dir(RAW_ROOT, TASK, jr[0].job_id)
        if job_dir is None:
            raise SystemExit(f"{job.label}: job {jr[0].job_id} has no raw data; recover it first")
        meta = json.loads((job_dir / "meta.json").read_text())
        if meta["calibration_set_id"] not in cals:
            raise SystemExit(f"{job.label} ran on {meta['calibration_set_id']}; pass its snapshot (have {sorted(cals)})")
        snap_path, cal = cals[meta["calibration_set_id"]]
        durations = dev.Durations.from_calibration(cal)
        tl = BR.timeline_numbers(meta.get("timeline") or [], shots=job.total_shots)
        job_out.append({"label": job.label, "job_id": meta["job_id"], "calibration_set_id": meta["calibration_set_id"],
                        "circuits": len(job.names), "shots": job.shots, "total_shots": job.total_shots,
                        "ledger_time": jr[0].time, "timeline": tl, "raw": _rel(job_dir, root)})
        log(f"{job.label} {meta['job_id']}: {len(job.names)} circuits, timeline {json.dumps(tl)}")
        labels = np.full(job.shots, SPLIT_LABEL, dtype=object)
        job_checks = []
        for i, name in enumerate(job.names):
            spec, entry = fz.suite[name], fz.placement[name]
            raw, _ = D.load_memory(job_dir, i, circuit=spec.stim)
            rid = run_id(job.label, name)
            dec, prior_info, note = decoder_priors(cal, spec)
            files = {"stim": f"circuits/{spec.experiment}/{name}.stim", "qasm": f"circuits/{spec.experiment}/{name}.qasm",
                     "qasm_sha256": entry["qasm_sha256"], "native_sha256": entry["native_sha256"]}
            ds, numbers = IR.write_dataset(
                root, spec, raw, run_id=rid, batch=BATCH, job_id=meta["job_id"], position_in_job=i, arch=fz.suite.arch,
                cal=cal, durations=durations, circuit_files=files,
                raw_ref={"path": _rel(job_dir / "memory.npz", root), "key": f"c{i}", "meta": _rel(job_dir / "meta.json", root)},
                calibration_paths={**snap_rel, "job": _rel(snap_path, root)}, decoder_circuit=dec, decoder_note=note,
                split_labels=labels, split_info=split_info(job.label, job.shots), simulated=False,
                extra={"device": DEVICE, "job_timeline": meta.get("timeline")},
            )
            der, _summary = IR.write_derived(root, spec, rid, ds)
            tag_json(der / "summary.json", label="HARDWARE", device=DEVICE, batch=BATCH, split=SPLIT_LABEL, priors=PRIORS)
            problems = IR.check_run(ds, der)
            dets = IR.load_array(ds, "dets")
            with open(ds / "detectors.csv") as f:
                t = np.array([int(r["round"]) for r in csv.DictReader(f)])
            ref = BR.reference(DEVICE, spec, cal)
            checks = BR.sanity(raw, dets, ref, shots=job.shots, physical=(meta.get("physical_qubits") or [None] * 99)[i],
                               planned=fz.planned_qubits(name))
            body = {"format": IR.FORMAT, "label": "HARDWARE", "task": TASK, "device": DEVICE, "batch": BATCH,
                    "split": SPLIT_LABEL, "job": job.label, "job_id": meta["job_id"], "position_in_job": i, "circuit": name,
                    "calibration_set_id": meta["calibration_set_id"], "priors": PRIORS, "prior": prior_info,
                    "sanity": checks, "sanity_pass": all(c["pass"] for c in checks),
                    "per_round_detection_rate": BR.per_round(dets, t) if dets.size else {},
                    "model_with_idle": ref["model"], "numbers": numbers, "record_problems": problems,
                    "created_utc": utc_now(), "commit": hw.git_head()}
            write_json(der / "fresh_checks.json", body)
            job_checks.append({"pass": body["sanity_pass"], "checks": checks})
            runs.append({"job": job.label, "spec": spec, "run_id": rid, "dataset": ds, "derived": der, "numbers": numbers,
                         "problems": problems, "sanity_pass": body["sanity_pass"], "note": note})
            if problems or not body["sanity_pass"]:
                log(f"  {rid}: problems {problems}; sanity {body['sanity_pass']}")
        job_out[-1]["sanity"] = BR.sanity_line(job_checks)
        job_out[-1]["detection_ratio_to_model"] = [c["info"]["ratio_to_model"] for jc in job_checks for c in jc["checks"]
                                                   if c["check"] == "detection fraction" and c.get("info", {}).get("ratio_to_model")]

    def point(r: Mapping[str, Any]) -> dict[str, Any]:
        return {"circuit": r["spec"].name, "dataset": _rel(r["dataset"], root), "rounds": r["spec"].rounds,
                "shots": r["numbers"]["shots"], "failures": r["numbers"]["pm_failures"], "raw_failures": r["numbers"]["obs_flips"]}

    ler, fresh_points = {}, {}
    for label, (_src, basis) in SOURCE.items():
        rs = sorted((r for r in runs if r["job"] == label), key=lambda r: r["spec"].rounds)
        pts = [point(r) for r in rs]
        rid = f"fresh_{basis}_{PATCH}"
        path, res = IR.write_ler(root, "E1", rid, pts, title=f"HARDWARE ({DEVICE}): E1 d=3 {basis} basis, patch {PATCH}, fresh "
                                 f"({label}, {TASK})", label=rid, fit=True)
        tag_json(path / "ler_fits.json", label="HARDWARE", device=DEVICE, batch=BATCH, split=SPLIT_LABEL, priors=PRIORS,
                 decoder=DECODER, noise=rs[0]["note"], task=TASK, job=label,
                 zero_failure_points=[x["rounds"] for x in pts if x["failures"] == 0])
        ler[basis] = {"record": _rel(path, root), "job": label}
        fresh_points[basis] = pts
    f0 = next(r for r in runs if r["job"] == "F0")
    pilot = point(f0) | {"wilson95": list(stats.wilson(f0["numbers"]["pm_failures"], f0["numbers"]["shots"]))}

    batch = BR.read_points(root, "E1")
    out_fits: dict[str, Any] = {}
    compare: dict[str, Any] = {}
    for basis in ("Z", "X"):
        fp = [{**p, "basis": basis} for p in fresh_points[basis]]
        bp = [p for p in batch if p["basis"] == basis and p["patch"] == PATCH]
        b3 = [p for p in bp if p["batch"] == 3]
        b12 = [p for p in bp if p["batch"] in (1, 2)]
        out_fits[basis] = {name: {"all_R": fits(pts), "R_le_3": fits(pts, 3)}
                           for name, pts in (("fresh", fp), ("batch_b3", b3), ("batch_b1b2", b12))}
        by_r = {p["rounds"]: p for p in b3}
        compare[basis] = [{"rounds": p["rounds"], "fresh": [p["failures"], p["shots"]],
                           "b3": [by_r[p["rounds"]]["failures"], by_r[p["rounds"]]["shots"]],
                           **two_proportions(p["failures"], p["shots"], by_r[p["rounds"]]["failures"], by_r[p["rounds"]]["shots"])}
                          for p in fp if p["rounds"] in by_r]
    return {"format": IR.FORMAT, "label": "HARDWARE", "task": TASK, "device": DEVICE, "split": SPLIT_LABEL, "batch": BATCH,
            "priors": PRIORS, "decoder": DECODER, "jobs": job_out, "runs": len(runs),
            "record_problems": {r["run_id"]: r["problems"] for r in runs if r["problems"]},
            "sanity_failures": [r["run_id"] for r in runs if not r["sanity_pass"]],
            "ler_records": ler, "points": fresh_points, "pilot_point": pilot, "fits": out_fits, "per_round_vs_b3": compare,
            "b3_brief": B3_EPS, "fit_model": "2 P_L = 1 - A (1 - 2 eps_L)^R, binomial maximum likelihood (dqiqm.stats.fit_eps); "
                                              "A = 1 is the challenge's formula", "bootstrap": BOOTSTRAP, "seed": SEED,
            "snapshots": [_rel(Path(p), REPO) for p in snapshots], "created_utc": utc_now(), "commit": hw.git_head()}


def cmd_analyze(args: argparse.Namespace) -> int:
    res = analyze([Path(p) for p in args.snapshots], root=RECORD, ledger=Path(args.ledger))
    out = SUMMARY_DIR / "fresh_summary.json"
    write_json(out, res)
    with open(SUMMARY_DIR / "fresh_points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["basis", "rounds", "shots", "failures", "raw_failures", "circuit", "dataset"],
                           lineterminator="\n")
        w.writeheader()
        for basis, pts in res["points"].items():
            for p in pts:
                w.writerow({"basis": basis, **p})
    for basis, f in res["fits"].items():
        for pool, x in f.items():
            for span, y in x.items():
                if "fixed_A" in y:
                    log(f"{basis} {pool} {span}: eps(A=1) {y['fixed_A']['eps']:.4f} {y['fixed_A']['eps_ci_bootstrap']}, "
                        f"eps(free A) {y['free_A']['eps']:.4f} A {y['free_A']['amplitude']:.3f}")
    log(f"ANALYZED {TASK}: {res['runs']} runs; record problems {res['record_problems'] or 'none'}; "
        f"sanity failures {res['sanity_failures'] or 'none'}; wrote {_rel(out, REPO)}")
    return 0 if not res["record_problems"] and not res["sanity_failures"] else 1


# ---------------------------------------------------------------------------
# doc
# ---------------------------------------------------------------------------


def _ci(ci: Sequence[float] | None, nd: int = 4) -> str:
    return "n/a" if not ci or ci[0] is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def doc_lines(s: Mapping[str, Any]) -> list[str]:
    out = ["# emerald-fresh: fresh Emerald shots, an independent check of the frozen decoders", "",
           "Sal (2026-10-25 02:49 UTC): \"Run with fresh emerald shots directly from the cloud backend aka the api\". "
           "Three jobs on IQM Emerald through IQM Resonance's cloud API (`dqiqm.hw.submit`, DD off), taken after every "
           "decoder was trained, tuned and selected. **Nothing is chosen on these shots.** Their split is `fresh`: never "
           "finetune, validation or test, and never in the ML training or tuning data. Driver: `scripts/emerald_fresh.py`; "
           f"generated by its `doc` command from `{_rel(SUMMARY_DIR / 'fresh_summary.json', REPO)}`.", "",
           f"Decoder: {s['decoder']}, priors **{s['priors']}**.", "",
           "## The jobs", "",
           "| job | job id | calibration set | circuits x shots | ledger (UTC) | queue (s) | execution span (s) | span per shot (us) | "
           "sanity | detection / model |", "|---|---|---|---|---|---|---|---|---|---|"]
    for j in s["jobs"]:
        tl, ratios = j["timeline"], j.get("detection_ratio_to_model") or []
        q = tl.get("queue_s")
        out.append(f"| {j['label']} | `{j['job_id']}` | `{j['calibration_set_id'][:8]}` | {j['circuits']} x {j['shots']:,} | "
                   f"{j['ledger_time']} | {'n/a' if q is None else f'{q:.1f}'} | {tl.get('execution_s', float('nan')):.2f} | "
                   f"{tl.get('execution_per_shot_us', float('nan')):.1f} | {j.get('sanity', '')} | "
                   f"{f'{min(ratios):.2f}-{max(ratios):.2f}' if ratios else 'n/a'} |")
    out += ["", "Timelines (first timestamp of each status, UTC):", ""]
    for j in s["jobs"]:
        tl = j["timeline"]
        out.append(f"- {j['label']}: created {tl.get('created')}, execution {tl.get('execution_started')} .. "
                   f"{tl.get('execution_ended')}, completed {tl.get('completed')} (total {tl.get('total_s')} s)")
    out += ["", f"Record problems: {s['record_problems'] or 'none'}. Sanity failures: {s['sanity_failures'] or 'none'}. "
            f"Snapshots: {', '.join(f'`{x}`' for x in s['snapshots'])}.", "",
            "## P_L(R), fresh (MWPM, Wilson 95%)", "",
            "| R | Z: failures / shots | Z: P_L [Wilson 95%] | X: failures / shots | X: P_L [Wilson 95%] |", "|---|---|---|---|---|"]
    z = {p["rounds"]: p for p in s["points"]["Z"]}
    x = {p["rounds"]: p for p in s["points"]["X"]}
    for r in sorted(set(z) | set(x)):
        cells = []
        for pts in (z, x):
            p = pts.get(r)
            if p is None:
                cells += ["", ""]
                continue
            lo, hi = stats.wilson(p["failures"], p["shots"])
            cells += [f"{p['failures']} / {p['shots']}", f"{p['failures'] / p['shots']:.4f} [{lo:.4f}, {hi:.4f}]"]
        out.append(f"| {r} | " + " | ".join(cells) + " |")
    pp = s["pilot_point"]
    out += ["", f"The pilot F0 (`{pp['circuit']}`, not pooled into the fits): {pp['failures']} / {pp['shots']} = "
            f"{pp['failures'] / pp['shots']:.4f} {_ci(pp['wilson95'])}.", "",
            "## eps_L per round", "",
            "2 P_L = 1 - A (1 - 2 eps_L)^R, binomial maximum likelihood; A = 1 is the challenge's fit. Intervals are bootstrap "
            f"95% ({s['bootstrap']} draws, seed {s['seed']}). The batch's pools are re-fitted here from its own records "
            "(`InfoRecord/emerald/ler/E1/*_eA_b[123]`), with the same prior.", "",
            "| basis | pool | R | shots | eps_L, A = 1 [bootstrap 95%] | chi2/dof | eps_L, A free [bootstrap 95%] | A |",
            "|---|---|---|---|---|---|---|---|"]
    for basis in ("X", "Z"):
        for pool, spans in s["fits"][basis].items():
            for span, f in spans.items():
                if "fixed_A" not in f:
                    continue
                fa, fr = f["fixed_A"], f["free_A"]
                out.append(f"| {basis} | {pool} | {'all' if span == 'all_R' else '<= 3'} | {sum(f['shots']):,} | "
                           f"{'**' if pool == 'fresh' else ''}{fa['eps']:.4f}{'**' if pool == 'fresh' else ''} "
                           f"{_ci(fa['eps_ci_bootstrap'])} | {fa['chi2_per_dof']:.2f} | {fr['eps']:.4f} "
                           f"{_ci(fr['eps_ci_bootstrap'])} | {fr['amplitude']:.3f} |")
    out += ["", "## Fresh against the batch's B3, per R", "",
            "Two-proportion z per R (information; no correction for the 11 comparisons per basis).", "",
            "| basis | R | fresh | B3 | P_L fresh - B3 | z | p |", "|---|---|---|---|---|---|---|"]
    for basis in ("X", "Z"):
        for c in s["per_round_vs_b3"][basis]:
            out.append(f"| {basis} | {c['rounds']} | {c['fresh'][0]}/{c['fresh'][1]} | {c['b3'][0]}/{c['b3'][1]} | "
                       f"{c['diff']:+.4f} | {c['z']:+.2f} | {c['p_two_sided']:.3f} |")
    out += ["", "## LSTM (frozen, decoded on the laptop)", "",
            "_Placeholder for the orchestrator: the frozen LSTM's decode of these fresh datasets (transfer 5)._", "",
            "## How to reproduce", "", "```",
            "uv sync --locked && uv run pytest tests/test_emerald_fresh.py",
            "uv run python scripts/emerald_fresh.py plan",
            "uv run python scripts/emerald_fresh.py verify                      # offline: the freeze and the references",
            "# the records are written once; to re-run analyze, do it in a fresh checkout with the fresh_* records removed",
            "flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\",
            "    uv run python scripts/emerald_fresh.py analyze --snapshots <before> <after>",
            "uv run python scripts/emerald_fresh.py doc", "```", ""]
    return out


def cmd_doc(args: argparse.Namespace) -> int:
    s = json.loads((SUMMARY_DIR / "fresh_summary.json").read_text())
    notes = Path(args.notes).read_text().rstrip() + "\n\n" if args.notes else ""
    lines = doc_lines(s)
    DOC.write_text("\n".join(lines[:2]) + "\n" + notes + "\n".join(lines[2:]) + "\n")
    log(f"wrote {_rel(DOC, REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="offline: the allowance against the job plan")
    sub.add_parser("snapshot", help="live, read-only: one Emerald snapshot")
    v = sub.add_parser("verify", help="offline freeze; with --snapshot the live checks; with --live the read-only requests")
    v.add_argument("--snapshot", type=Path)
    v.add_argument("--live", action="store_true")
    v.add_argument("--out", type=Path)
    s = sub.add_parser("submit", help="the jobs given, in order (armed by DQ_QPU=run ...)")
    s.add_argument("--snapshot", type=Path, required=True, help="the before snapshot (its calibration set is used)")
    s.add_argument("--jobs", nargs="+", choices=SEG.labels, required=True)
    s.add_argument("--timeout", type=float, default=None)
    r = sub.add_parser("recover", help="read-only: re-pull a submitted job and save it as save_raw does")
    r.add_argument("--job-id", required=True)
    r.add_argument("--snapshot", type=Path)
    r.add_argument("--timeout", type=float, default=None)
    a = sub.add_parser("analyze", help="offline: the records, fits and comparison")
    a.add_argument("--snapshots", type=Path, nargs="+", required=True, help="before (first), any mid-run, after (last)")
    a.add_argument("--ledger", type=Path, default=Path(hw.DEFAULT_LEDGER))
    d = sub.add_parser("doc", help="offline: docs/emerald-fresh.md from the summary")
    d.add_argument("--notes", type=Path, default=None)
    args = ap.parse_args()
    return {"plan": cmd_plan, "snapshot": cmd_snapshot, "verify": cmd_verify, "submit": cmd_submit,
            "recover": cmd_recover, "analyze": cmd_analyze, "doc": cmd_doc}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
