"""The Garnet hardware job plan and the pilot's sanity bands, written into docs/garnet-design.md.

    uv run python scripts/garnet_plan.py [--no-write]

Builds the suite from the committed live snapshot (no network), lists every job with its circuits,
shots and estimated QPU time (``dqiqm.experiments.suite``), and computes what the pilot (E1, Z,
R = 3, patch pA) should look like:

* on the stand-in, from the dry run's P0 dataset (``InfoRecord/sim``) when it exists;
* under the calibrated Stim noise with the readout idle as the pilot's band was built (the 556 ns probe,
  a symmetric idle, the idle after the last round: the band the pilot passed, which ``garnet_pilot.py``
  re-reads), and under the hardware prior since garnet-timing, ``device.hardware_noise`` (the compiled
  window as a symmetric idle, ``measure_fidelity``'s errors on the final layer), for reference.

Each job's QPU time charges the conservative per-shot overhead, the reset wait. The pilot's
execution span (``PILOT_CHECKS``) is recorded beside it, not planned on: it does not bracket the shot
loop (``orchestration/reports/pilots-review.md``).

The tables replace the text between ``<!-- plan:begin -->`` and ``<!-- plan:end -->`` in
``docs/garnet-design.md``, and the plan goes to ``InfoRecord/circuits/job_plan.json``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from dqiqm import inforecord as IR
from dqiqm import stats
from dqiqm.experiments import device as dev
from dqiqm.experiments import suite as SU

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "garnet-design.md"
#: The pilot's record (garnet-pilot): its job timeline's execution span.
PILOT_CHECKS = REPO / "InfoRecord" / "derived" / "E1" / "p0_E1_d3_Z_r03_pA" / "pilot_checks.json"
BEGIN, END = "<!-- plan:begin -->", "<!-- plan:end -->"


def round1_cols(spec) -> tuple[list[int], list[int]]:
    """The raw_meas columns (clbit j = Stim measurement j) of the X-type and the Z-type ancillas' round-1 bits."""
    meas_q = IR.C.measured_qubits(spec.stim)
    first = [j for j, q in enumerate(meas_q) if spec.meas_round[j] == 1]
    return ([j for j in first if meas_q[j] in spec.x_ancillas], [j for j in first if meas_q[j] not in spec.x_ancillas])


def pilot_profile(spec, raw: np.ndarray, dets: np.ndarray) -> dict:
    """The pilot's sanity numbers from a record: detection fraction, round-1 ancilla bits, all-zero rows."""
    x_cols, z_cols = round1_cols(spec)
    n = len(raw)
    k = int(dets.sum())
    return {
        "shots": n,
        "columns": raw.shape[1],
        "detection_fraction": k / dets.size,
        "detection_fraction_ci": list(stats.wilson(k, dets.size)),
        "x_ancilla_m1_one": float(raw[:, x_cols].mean()),
        "z_ancilla_m1_one": float(raw[:, z_cols].mean()),
        "all_zero_rows": int((~raw.any(axis=1)).sum()),
    }


def pilot_expectations(suite) -> dict:
    spec = next(s for s in suite.of("E1") if s.params["basis"] == "Z" and s.rounds == 3 and s.params["patch"] == "pA"
                and s.params["barriers"])
    out = {"circuit": spec.name}
    ds = REPO / "InfoRecord" / "sim" / "datasets" / "E1" / f"p0_{spec.name}"
    if ds.exists():
        out["stand_in"] = pilot_profile(spec, IR.load_array(ds, "raw_meas"), IR.load_array(ds, "dets"))
        out["stand_in"]["source"] = str(ds.relative_to(REPO))
    args = dict(pairs=spec.cx_pairs, x_ancillas=spec.x_ancillas)
    bare = dev.calibrated_noise(suite.cal, spec.physical, spec.roles, **args)
    # The band is the one the pilot was checked against (scripts/garnet_pilot.py re-reads it), so it keeps the priors
    # it was built from: the 556 ns measure probe as the window, a symmetric idle, and the idle after the last round.
    anc = [spec.physical[a] for a in spec.stabilizers]
    idle = max(suite.durations.measure_ns[q] for q in anc)
    old = dataclasses.replace(dev.calibrated_noise(suite.cal, spec.physical, spec.roles, idle_ns=idle, **args),
                              idle_after_last_round=True)
    hard, info = dev.hardware_noise(suite.cal, spec)
    for name, nm, extra in (("calibrated_stim_no_idle", bare, {"idle_ns": 0.0}),
                            ("calibrated_stim_with_idle", old, {"idle_ns": idle}),
                            ("hardware_prior", hard, {"idle_ns": info["idle_ns"], "prior": info})):
        noisy = spec.noisy(nm)
        raw = noisy.compile_sampler(seed=2026).sample(100_000).astype(np.uint8)
        dets, _ = IR.D.detection_events(spec.stim, raw)
        out[name] = pilot_profile(spec, raw, dets) | extra
    out["hardware_prior"]["note"] = ("garnet-timing's hardware prior (device.hardware_noise), for reference; the band below "
                                     "stays the one the pilot passed, built from calibrated_stim_with_idle (the pre-fix priors)")
    hard_prior = out.pop("hardware_prior")
    hi = out["calibrated_stim_with_idle"]["detection_fraction"]
    lo = out.get("stand_in", out["calibrated_stim_no_idle"])["detection_fraction"]
    # a broken readout or bit order gives detection near 0.5, all-zero rows near 100%, or X-type round-1 bits far from 0.5
    x_cols, z_cols = round1_cols(spec)
    out["band"] = {"detection_fraction": [0.5 * lo, 2.0 * hi], "x_ancilla_m1_one": [0.4, 0.6],
                   "z_ancilla_m1_one_max": 0.3, "all_zero_rows_max_per_1000": 5, "columns": spec.stim.num_measurements,
                   "x_cols": x_cols, "z_cols": z_cols,
                   "note": "x_ancilla_m1_one and z_ancilla_m1_one are pooled means: the fraction of 1s over every shot and "
                           "every column in x_cols (z_cols) of raw_meas, clbit j = column j. detection_fraction is the "
                           "detection events over shots x detectors."}
    out["hardware_prior"] = hard_prior
    return out


def pilot_duration(suite) -> float:
    s = next(x for x in suite.of("E1") if x.params["basis"] == "Z" and x.rounds == 3 and x.params["patch"] == "pA")
    return SU.schedule(s.hardware, suite.durations).duration_ns / 1e3


def pilot_span() -> dict | None:
    """The pilot's execution span over its shots, for the record: not a planning figure."""
    if not PILOT_CHECKS.exists():
        return None
    rec = json.loads(PILOT_CHECKS.read_text())
    return {"job_id": rec["job_id"], "execution_s": float(rec["timeline_numbers"]["execution_s"]),
            "per_shot_s": float(rec["timeline_numbers"]["per_shot_s"]), "source": str(PILOT_CHECKS.relative_to(REPO)),
            "planning": False,
            "note": "execution_started to execution_ended over 1,000 shots: less than the 298 us reset wait alone. The raw "
                    "shots show an inter-shot wait of about 200 us or more (round-1 Z-type mean 0.079, the perfect-start "
                    "value; orchestration/reports/pilots-review.md), so the span does not bracket the shot loop. Plan on "
                    "reset_wait per shot."}


def plan_table(plan: list[dict]) -> str:
    lines = ["| job | phase | exp | circuits | shots / circuit | total shots | circuit (µs), min–max (mean) | per shot (µs) = mean + reset_wait | "
             "shots × per shot (s) | + circuits × 0.1 s | **est. QPU (s)** | pessimistic (s) | what |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in plan:
        per = j["circuit_us_mean"] + j["per_shot_overhead_s"] * 1e6
        lines.append(f"| {j['label']} | {j['phase']} | {j['experiment']} | {j['circuits']} | {j['shots']:,} | {j['total_shots']:,} | "
                     f"{j['circuit_us_min']:.1f}–{j['circuit_us_max']:.1f} ({j['circuit_us_mean']:.1f}) | {per:.1f} | "
                     f"{j['qpu_s_shots']:.1f} | {j['qpu_s_per_circuit_overhead']:.1f} | **{j['qpu_s']:.1f}** | {j['qpu_s_pessimistic']:.0f} | {j['note']} |")
    tot_shots = sum(j["total_shots"] for j in plan)
    lines.append(f"| **total** | | | {sum(j['circuits'] for j in plan)} | | **{tot_shots:,}** | | | {sum(j['qpu_s_shots'] for j in plan):.0f} | "
                 f"{sum(j['qpu_s_per_circuit_overhead'] for j in plan):.0f} | **{sum(j['qpu_s'] for j in plan):.0f} s "
                 f"({sum(j['qpu_s'] for j in plan) / 60:.1f} min)** | **{sum(j['qpu_s_pessimistic'] for j in plan):.0f}** | |")
    return "\n".join(lines)


def per_experiment(plan: list[dict]) -> str:
    by: dict[str, list] = {}
    for j in plan:
        by.setdefault(j["experiment"], []).append(j)
    lines = ["| experiment | jobs | circuit runs | shots | est. QPU (s) |", "|---|---|---|---|---|"]
    for e, js in sorted(by.items()):
        lines.append(f"| {e} | {len(js)} | {sum(j['circuits'] for j in js)} | {sum(j['total_shots'] for j in js):,} | "
                     f"{sum(j['qpu_s'] for j in js):.0f} |")
    return "\n".join(lines)


def split_table(plan: list[dict]) -> str:
    e1 = [j for j in plan if j["experiment"] == "E1"]
    test = sum(j["total_shots"] for j in e1 if j["batch"] == SU.SPLIT["test_batch"])
    rest = sum(j["total_shots"] for j in e1 if j["batch"] != SU.SPLIT["test_batch"])
    val = round(SU.SPLIT["validation_fraction"] * rest)
    return (f"| E1 split | shots |\n|---|---|\n| finetune (batches 1, 2 and the pilot P0, 80%) | {rest - val:,} |\n"
            f"| validation (the same jobs, 20% of shot indices) | {val:,} |\n| test (batch {SU.SPLIT['test_batch']}, whole) | {test:,} |\n"
            f"| **total E1** | **{test + rest:,}** |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()
    cal = dev.live_calibration()
    suite = SU.build(cal)
    jobs = SU.jobs(suite)
    span = pilot_span()
    plan = [SU.estimate(j, suite) for j in jobs]
    pilot = pilot_expectations(suite)
    md = "\n".join([
        BEGIN,
        f"Generated by `scripts/garnet_plan.py` from calibration set `{cal.calibration_set_id}`.",
        "",
        f"**Estimated QPU time of a job = Σ over its circuits of shots × (circuit duration + reset_wait) + circuits × "
        f"{SU.PER_CIRCUIT_OVERHEAD_S} s.** reset_wait is the device's passive-reset wait, {cal.reset_wait_ns / 1e3:.0f} µs "
        f"(`gates.reset_wait` in the snapshot): the qubits relax between shots, as there is no active reset. Any other per-shot "
        f"cost is still unknown (the pilot's timeline did not settle it, below); the 'pessimistic' column instead charges "
        f"{SU.PESSIMISTIC_PER_SHOT_S * 1e3:.2f} ms per shot, the whole per-shot cost implied by Garnet's published CLOPS of "
        f"2,600 (100 × 10 × 100 × 5 / 2,600 s over 10⁵ shots). Circuit durations come from the calibrated durations (prx "
        f"{cal.median('prx_ns'):.0f} ns, CZ {cal.median('cz_ns', couplers=True):.0f} ns, measure {cal.median('measure_ns'):.0f} ns "
        f"median; each readout layer takes its slowest qubit's) with barriers as sync points (`circuits.schedule`): e.g. E1 "
        f"R = 3 is {pilot_duration(suite):.2f} µs. That model counts a readout as its `measure` probe; IQM's compiler makes each "
        "readout window about twice as long (pulla-check: the pilot compiles to 3.90 µs), which moves a shot by a few µs "
        "against the 298 µs reset wait.",
        "",
        *([f"**The pilot's timeline is not a planning figure.** Job `{span['job_id']}`'s execution span is "
           f"{span['execution_s'] * 1e3:.2f} ms for 1,000 shots, {span['per_shot_s'] * 1e6:.1f} µs per shot "
           f"(`{span['source']}`), against the {SU.per_shot_overhead_s(cal) * 1e6:.0f} µs reset wait planned here. That is "
           "less than the wait alone, yet the raw shots show the qubits did wait about 200 µs or more between shots (the round-1 "
           "Z-type mean is 0.079, the perfect-start value; a 48 µs wait would give about 0.34: `orchestration/reports/pilots-review.md`). "
           "So the span does not bracket the shot loop, and the plan charges the reset wait per shot.", ""] if span else []),
        plan_table(plan),
        "",
        per_experiment(plan),
        "",
        split_table(plan),
        "",
        "**Pilot expectations** (E1 Z R=3 pA; 'band' is what the pilot must fall in before anything else runs):",
        "",
        "```json",
        json.dumps(pilot, indent=1),
        "```",
        END,
    ])
    print(md)
    if args.no_write:
        return 0
    (REPO / "InfoRecord" / "circuits").mkdir(parents=True, exist_ok=True)
    (REPO / "InfoRecord" / "circuits" / "job_plan.json").write_text(json.dumps(
        SU.plan_json(suite, plan, pilot_span=span) | {"pilot": pilot}, indent=1, default=str) + "\n")
    text = DOC.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{DOC} has no {BEGIN} ... {END} section")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(head + md + tail)
    print(f"wrote {DOC.relative_to(REPO)} and InfoRecord/circuits/job_plan.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
