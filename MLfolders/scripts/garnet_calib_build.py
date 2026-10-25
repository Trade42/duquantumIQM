"""Build the Garnet characterization circuits, check them, and write the record and the job plan.

    uv run python scripts/garnet_calib_build.py             # writes InfoRecord/characterization/{circuits,job_plan.json}
    uv run python scripts/garnet_calib_build.py --check     # rebuilds and compares with the committed record, writes nothing
    uv run python scripts/garnet_calib_build.py --plan-md   # prints the plan table for docs/garnet-calib.md

Offline, no token, about 20 s. Every circuit is placed at optimization level 0 on IQMFakeGarnet
(the live calibration's architecture), proved by ``characterization.circuits.check`` (native ops,
couplers, the designed per-qubit sequence, a barrier between every two ops, ``count_ops`` equal to
the design) and by IQM's ``validate_circuit``. Nothing is submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dqiqm import layout as L  # noqa: E402
from dqiqm.characterization import circuits as CC  # noqa: E402
from dqiqm.characterization import leakage as LK  # noqa: E402
from dqiqm.characterization import patch as P  # noqa: E402
from dqiqm.characterization import plan as PL  # noqa: E402
from dqiqm.characterization import pulse as PU  # noqa: E402
from dqiqm.experiments import device as dev  # noqa: E402

RECORD = REPO / "InfoRecord" / "characterization"
CIRCUITS = RECORD / "circuits"
FORMAT = "dqiqm.characterization/1"


def backend():
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    return IQMFakeGarnet()


def live_couplers(snapshot: dict) -> set[frozenset]:
    arch = snapshot["architecture"]["gates"]["cz"]["implementations"]
    return {frozenset(locus) for impl in arch.values() for locus in impl["loci"]}


def build(log=print) -> dict:
    """Everything the record holds, in memory."""
    cal = dev.live_calibration()
    snapshot = json.loads(P.SNAPSHOT.read_text())
    be = backend()
    arch = L.Architecture.from_backend(be)
    patch = P.pilot_patch(be, cal)
    pri = P.priors(patch, cal, snapshot)
    live = live_couplers(snapshot)
    missing = [p for p in patch.couplers if frozenset(p) not in live]
    if missing:
        raise AssertionError(f"patch couplers without a CZ in the live calibration: {missing}")
    if {frozenset(c) for c in arch.couplers} != live:
        raise AssertionError("IQMFakeGarnet's couplers differ from the live calibration's")
    designs = CC.c1_designs(patch) + CC.c2_designs(patch) + CC.c2_pilot_designs(patch) + CC.c4_designs(patch)
    durations = {"prx": max(v["prx_ns"] for v in pri.qubits.values()),
                 "cz": max(v["cz_ns"] for v in pri.couplers.values())}
    entries: dict[str, dict] = {}
    files: dict[str, str] = {}
    for d in designs:
        p = CC.place(d, patch, be)
        sent = CC.check(p, arch, patch.couplers)
        CC.validate(p, be)
        text = CC.qasm(p)
        files[f"{d.experiment}/{d.name}.qasm"] = text
        entries[d.name] = {
            "experiment": d.experiment,
            "device": P.DEVICE,
            "calibration_set_id": pri.calibration_set_id,
            "target": "hardware",
            "params": dict(d.params),
            "layers": len(d.layers),
            "ops": dict(sorted(p.circuit.count_ops().items())),
            "sent": dict(sorted(sent.items())),
            "duration_ns": CC.duration_ns(d, durations, pri.terminal_window_ns),
            "native_sha256": CC.native_sha256(p),
            "qasm_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "checks": {"check": "passed", "validate_circuit": "passed (IQMFakeGarnet)", "swaps": 0},
        }
    log(f"built and checked {len(designs)} circuits: "
        + ", ".join(f"{e} {sum(1 for d in designs if d.experiment == e)}" for e in ("C1", "C2", "C4")))
    by_name = {d.name: d for d in designs}
    plan_jobs = PL.jobs(by_name, patch)
    per_shot = (cal.reset_wait_ns or 298_000) * 1e-9
    estimates = [PL.estimate(j, by_name, pri, durations, per_shot) for j in plan_jobs]
    bands = PL.pilot_bands(patch, pri)
    bands["circuits"] = list(plan_jobs[0].circuits)
    cz_loci = sorted(json.loads(P.PULSE.read_text())["gates"]["cz"]["slepian_crf"])
    c3 = LK.design(patch.couplers, cz_loci)
    plan = {
        "format": FORMAT,
        "device": P.DEVICE,
        "calibration_set_id": pri.calibration_set_id,
        "task": "garnet-calib",
        "status": "designed; nothing submitted. Runs only after Sal's yes, through dqiqm.hw.submit, one worker at a time.",
        "shots": PL.SHOTS,
        "per_shot_overhead_s": per_shot,
        "per_circuit_overhead_s": PL.PER_CIRCUIT_OVERHEAD_S,
        "pessimistic_per_shot_s": PL.PESSIMISTIC_PER_SHOT_S,
        "durations_ns": {**durations, "terminal_readout_window": pri.terminal_window_ns},
        "jobs": estimates,
        "totals": {phase: {"jobs": sum(1 for e in estimates if e["phase"] == phase),
                           "circuits": sum(e["circuits"] for e in estimates if e["phase"] == phase),
                           "shots": sum(e["total_shots"] for e in estimates if e["phase"] == phase),
                           "qpu_s": sum(e["qpu_s"] for e in estimates if e["phase"] == phase),
                           "qpu_s_pessimistic": sum(e["qpu_s_pessimistic"] for e in estimates if e["phase"] == phase)}
                   for phase in ("pilot", "must", "should", "if time")},
        "pilot": bands,
        "blocked": {"C3": {**LK.summary(patch.couplers, cz_loci), "circuits": len(c3)}},
        "submission": {
            "how": "hw.submit([placed circuits in the job's order], device='garnet', shots=<job shots>, "
                   "qubit_index_to_name=<placement.json qubit_index_to_name>, calibration_set_id=<the backend's>, "
                   "purpose='garnet-calib <label>')",
            "before": "rebuild with scripts/garnet_calib_build.py --check against the live calibration; every "
                      "native_sha256 must equal placement.json's; validate against hw.backend('garnet')",
        },
    }
    common = {"column_qubits": list(patch.qubits),
              "qubit_index_to_name": {str(k): v for k, v in sorted(p.qubit_index_to_name.items())}}
    return {"patch": patch, "priors": pri, "entries": entries, "files": files, "plan": plan, "c3": c3, "common": common}


def write(out: dict, log=print) -> None:
    patch, pri = out["patch"], out["priors"]
    CIRCUITS.mkdir(parents=True, exist_ok=True)
    (CIRCUITS / "patch.json").write_text(json.dumps({"format": FORMAT, **patch.to_json(), "priors": pri.to_json()},
                                                    indent=1) + "\n")
    for exp in ("C1", "C2", "C4"):
        d = CIRCUITS / exp
        d.mkdir(exist_ok=True)
        for old in d.glob("*.qasm"):
            old.unlink()
        ent = {k: v for k, v in out["entries"].items() if v["experiment"] == exp}
        doc = {"format": FORMAT, "device": P.DEVICE, "calibration_set_id": pri.calibration_set_id, **out["common"],
               "note": "clbit j of every circuit is column_qubits[j]; submit with qubit_index_to_name",
               "circuits": ent}
        (d / "placement.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    for rel, text in out["files"].items():
        (CIRCUITS / rel).write_text(text)
    (CIRCUITS / "C3").mkdir(exist_ok=True)
    (CIRCUITS / "C3" / "design.json").write_text(json.dumps(
        {"format": FORMAT, "device": P.DEVICE, "calibration_set_id": pri.calibration_set_id,
         "status": out["plan"]["blocked"]["C3"]["status"],
         "note": "IQM native instructions (CircuitOperation name, locus, args); not runnable through dqiqm.hw today",
         "circuits": out["c3"]}) + "\n")
    (RECORD / "job_plan.json").write_text(json.dumps(out["plan"], indent=1) + "\n")
    (RECORD / "pulse_plan.json").write_text(json.dumps(PU.plan(patch), indent=1) + "\n")
    log(f"wrote {CIRCUITS.relative_to(REPO)} ({len(out['files'])} qasm) and {(RECORD / 'job_plan.json').relative_to(REPO)}")


def check_against_record(out: dict, log=print) -> int:
    bad = 0
    for exp in ("C1", "C2", "C4"):
        committed = json.loads((CIRCUITS / exp / "placement.json").read_text())["circuits"]
        built = {k: v for k, v in out["entries"].items() if v["experiment"] == exp}
        if set(committed) != set(built):
            log(f"{exp}: circuit names differ")
            bad += 1
        for name, e in built.items():
            c = committed.get(name, {})
            for key in ("native_sha256", "qasm_sha256"):
                if c.get(key) != e[key]:
                    log(f"{name}: {key} differs")
                    bad += 1
    log("record matches the rebuild" if not bad else f"{bad} differences")
    return 1 if bad else 0


def plan_md(plan: dict) -> str:
    head = ("| job | phase | circuits | shots / circuit | total shots | circuit (µs), min–max | est. QPU (s) "
            "| pessimistic (s) | what |\n|---|---|---|---|---|---|---|---|---|\n")
    rows = "".join(
        f"| {j['label']} | {j['phase']} | {j['circuits']} | {j['shots']:,} | {j['total_shots']:,} | "
        f"{j['circuit_us_min']:.2f}–{j['circuit_us_max']:.2f} | **{j['qpu_s']:.1f}** | {j['qpu_s_pessimistic']:.0f} | "
        f"{j['note']} |\n" for j in plan["jobs"])
    tot = "".join(
        f"| **{ph}** | | {t['circuits']} | | **{t['shots']:,}** | | **{t['qpu_s']:.1f}** | {t['qpu_s_pessimistic']:.0f} | "
        f"{t['jobs']} job(s) |\n" for ph, t in plan["totals"].items())
    return head + rows + tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare a rebuild with the committed record; write nothing")
    ap.add_argument("--plan-md", action="store_true", help="print the plan table (from the committed job_plan.json)")
    a = ap.parse_args()
    if a.plan_md:
        print(plan_md(json.loads((RECORD / "job_plan.json").read_text())))
        return 0
    out = build()
    if a.check:
        return check_against_record(out)
    write(out)
    b = out["plan"]["pilot"]
    print(f"pilot: {len(b['circuits'])} circuits x {b['shots']} shots; C1 no-merge D floor {b['c1_no_merge']['stop_if_below']:.4f} "
          f"(T1/T2-only prediction {b['c1_no_merge']['predicted_t1t2_only']:.4f}, both {b['c1_no_merge']['predicted_t1t2_and_rb']:.4f}, "
          f"sigma {b['c1_no_merge']['sigma_pooled']:.4f}); C2 no-merge R ceiling {b['c2_no_merge']['stop_if_above']:.4f} "
          f"(IRB prediction {b['c2_no_merge']['predicted_irb']:.4f}, sigma {b['c2_no_merge']['sigma_pooled']:.4f})")
    for ph, t in out["plan"]["totals"].items():
        print(f"{ph}: {t['jobs']} jobs, {t['circuits']} circuits, {t['shots']:,} shots, {t['qpu_s']:.1f} s est. QPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
