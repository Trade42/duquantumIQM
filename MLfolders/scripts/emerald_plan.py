"""The Emerald hardware job plan and the pilot's stop/go bands, into job_plan.json and docs/emerald-design.md.

    flock /home/sal/duquantumIQM/dq/run/sim.lock nice -n 15 taskset -c 12-15 \\
        uv run python scripts/emerald_plan.py [--no-write]

Run it after ``scripts/emerald_dryrun.py`` (the dry run never touches ``job_plan.json``, so the
order only decides whether the pilot's stand-in column is filled). It rebuilds the suite from the
committed snapshot (no network), checks every circuit's ``native_sha256`` against
``InfoRecord/emerald/circuits/<experiment>/placement.json`` (so the plan is for exactly the
recorded circuits), lists every job with its circuits, shots and estimated QPU time from Emerald's
own ``reset_wait`` and durations (``dqiqm.experiments.suite.estimate``), and computes what the
pilot (E1, Z, R = 3, patch eA) should look like:

* on the stand-in, from the dry run's EP0 dataset (``InfoRecord/emerald/sim``) when it exists;
* under the calibrated Stim noise, without idles and with the readout-window idle counted once
  (``dqiqm.experiments.emerald.calibrated_noise``), the hardware prior.

The pilot functions are ``scripts/garnet_plan.py``'s, with the band's columns named (garnet-design
review, MINOR 8).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dqiqm import inforecord as IR
from dqiqm import stats
from dqiqm.experiments import circuits as C
from dqiqm.experiments import emerald as EM
from dqiqm.experiments import suite as SU

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "emerald-design.md"
BEGIN, END = "<!-- plan:begin -->", "<!-- plan:end -->"


def pilot_columns(spec) -> tuple[list[int], list[int]]:
    """Round-1 clbits of the X-type and the Z-type ancillas (clbit j = Stim measurement j)."""
    meas_q = C.measured_qubits(spec.stim)
    first = [j for j, q in enumerate(meas_q) if spec.meas_round[j] == 1]
    return ([j for j in first if meas_q[j] in spec.x_ancillas], [j for j in first if meas_q[j] not in spec.x_ancillas])


def pilot_profile(spec, raw: np.ndarray, dets: np.ndarray) -> dict:
    """The pilot's sanity numbers from a record: detection fraction, round-1 ancilla bits, all-zero rows."""
    x_cols, z_cols = pilot_columns(spec)
    k = int(dets.sum())
    return {
        "shots": len(raw),
        "columns": raw.shape[1],
        "detection_fraction": k / dets.size,
        "detection_fraction_ci": list(stats.wilson(k, dets.size)),
        "x_ancilla_m1_one": float(raw[:, x_cols].mean()),
        "z_ancilla_m1_one": float(raw[:, z_cols].mean()),
        "all_zero_rows": int((~raw.any(axis=1)).sum()),
    }


def pilot_expectations(suite) -> dict:
    spec = suite[EM.pilot_name(suite)]
    x_cols, z_cols = pilot_columns(spec)
    out = {"circuit": spec.name, "device": EM.DEVICE,
           "x_cols": x_cols, "z_cols": z_cols,
           "x_qubits": [spec.physical[C.measured_qubits(spec.stim)[j]] for j in x_cols],
           "z_qubits": [spec.physical[C.measured_qubits(spec.stim)[j]] for j in z_cols]}
    ds = EM.RECORD / "sim" / "datasets" / "E1" / f"ep0_{spec.name}"
    if ds.exists():
        out["stand_in"] = pilot_profile(spec, IR.load_array(ds, "raw_meas"), IR.load_array(ds, "dets"))
        out["stand_in"]["source"] = str(ds.relative_to(REPO))
    idle = EM.readout_window_ns(spec, suite.durations)
    for name, idle_ns in (("calibrated_stim_no_idle", 0.0), ("calibrated_stim_with_idle", idle)):
        noisy = spec.noisy(EM.calibrated_noise(suite.cal, spec, idle_ns=idle_ns))
        raw = noisy.compile_sampler(seed=2026).sample(100_000).astype(np.uint8)
        dets, _ = IR.D.detection_events(spec.stim, raw)
        out[name] = pilot_profile(spec, raw, dets) | {"idle_ns": idle_ns}
    hi = out["calibrated_stim_with_idle"]["detection_fraction"]
    lo = out.get("stand_in", out["calibrated_stim_no_idle"])["detection_fraction"]
    # a broken readout or bit order gives detection near 0.5, all-zero rows near 100%, or X-type round-1 bits far from 0.5
    out["band"] = {
        "columns": spec.stim.num_measurements,
        "shots": 1000,
        "detection_fraction": [0.5 * lo, 2.0 * hi],
        "x_ancilla_m1_one": [0.4, 0.6],
        "z_ancilla_m1_one_max": 0.3,
        "all_zero_rows_max_per_1000": 5,
        "how": "memory.npz c0 is (shots, columns). x_ancilla_m1_one is the pooled mean of raw[:, x_cols] (all X-type "
               "round-1 bits of all shots together), z_ancilla_m1_one the pooled mean of raw[:, z_cols]. detection_fraction "
               "is the mean of decode.detection_events(<circuits/E1/<circuit>.stim>, raw)[0] over all shots and detectors. "
               "all_zero_rows counts shots whose 33 bits are all 0.",
    }
    return out


def check_recorded(suite) -> int:
    """Every circuit's native_sha256 equals the recorded placement.json's: the plan is for exactly those circuits."""
    n = 0
    for exp in ("E1", "E4"):
        path = EM.RECORD / "circuits" / exp / "placement.json"
        if not suite.of(exp):
            continue
        rec = json.loads(path.read_text())
        for s in suite.of(exp):
            if rec[s.name]["native_sha256"] != C.native_sha256(s.hardware):
                raise SystemExit(f"{s.name}: the rebuilt circuit differs from {path}; rerun scripts/emerald_dryrun.py --clean")
            n += 1
    return n


def plan_table(plan: list[dict]) -> str:
    lines = ["| job | phase | exp | circuits | shots / circuit | total shots | circuit (µs), min–max (mean) | per shot (µs) = mean + reset_wait | "
             "shots × per shot (s) | + circuits × 0.1 s | **est. QPU (s)** | pessimistic (s) | what |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in plan:
        per = j["circuit_us_mean"] + j["per_shot_overhead_s"] * 1e6
        lines.append(f"| {j['label']} | {j['phase']} | {j['experiment']} | {j['circuits']} | {j['shots']:,} | {j['total_shots']:,} | "
                     f"{j['circuit_us_min']:.1f}–{j['circuit_us_max']:.1f} ({j['circuit_us_mean']:.1f}) | {per:.1f} | "
                     f"{j['qpu_s_shots']:.1f} | {j['qpu_s_per_circuit_overhead']:.1f} | **{j['qpu_s']:.1f}** | {j['qpu_s_pessimistic']:.0f} | {j['note']} |")
    tot = sum(j["qpu_s"] for j in plan)
    lines.append(f"| **total** | | | {sum(j['circuits'] for j in plan)} | | **{sum(j['total_shots'] for j in plan):,}** | | | "
                 f"{sum(j['qpu_s_shots'] for j in plan):.0f} | {sum(j['qpu_s_per_circuit_overhead'] for j in plan):.0f} | "
                 f"**{tot:.0f} s ({tot / 60:.1f} min)** | **{sum(j['qpu_s_pessimistic'] for j in plan):.0f}** | |")
    return "\n".join(lines)


def split_table(plan: list[dict]) -> str:
    e1 = [j for j in plan if j["experiment"] == "E1"]
    test = sum(j["total_shots"] for j in e1 if j["batch"] == EM.SPLIT["test_batch"])
    rest = sum(j["total_shots"] for j in e1 if j["batch"] != EM.SPLIT["test_batch"])
    val = round(EM.SPLIT["validation_fraction"] * rest)
    return (f"| E1 split | shots |\n|---|---|\n| finetune (batches 1, 2 and the pilot, 80% of shot indices) | {rest - val:,} |\n"
            f"| validation (the same jobs, 20% of shot indices; load as finetune or validation, never as test) | {val:,} |\n"
            f"| test (batch {EM.SPLIT['test_batch']}, whole) | {test:,} |\n| **total E1** | **{test + rest:,}** |")


def pilot_md(p: dict) -> str:
    b = p["band"]
    si = p.get("stand_in")
    lines = [
        f"**The pilot (EP0: `{p['circuit']}`, 1 job, 1 circuit, 1,000 shots), before anything else.** Stop and report if any check fails.",
        f"1. `memory.npz` `c0` is (1000, {b['columns']}), and the `calibration_set_id` is recorded.",
        f"2. At most {b['all_zero_rows_max_per_1000']} all-zero rows."
        + (f" The stand-in gives {si['all_zero_rows']}," if si else "")
        + " the priors {:.2f}–{:.2f} per 1,000.".format(*sorted(p[k]["all_zero_rows"] / 100 for k in
                                                              ("calibrated_stim_no_idle", "calibrated_stim_with_idle"))),
        f"3. The X-type ancillas' round-1 bits, clbits `x_cols` = {p['x_cols']} ({', '.join(p['x_qubits'])}), pooled mean in "
        f"[{b['x_ancilla_m1_one'][0]}, {b['x_ancilla_m1_one'][1]}]."
        + (f" The stand-in gives {si['x_ancilla_m1_one']:.3f}," if si else "")
        + f" the priors {p['calibrated_stim_no_idle']['x_ancilla_m1_one']:.3f}.",
        f"4. The Z-type ancillas' round-1 bits, clbits `z_cols` = {p['z_cols']} ({', '.join(p['z_qubits'])}), pooled mean under "
        f"{b['z_ancilla_m1_one_max']}." + (f" The stand-in gives {si['z_ancilla_m1_one']:.3f}," if si else "")
        + f" the priors {p['calibrated_stim_no_idle']['z_ancilla_m1_one']:.3f} without idle, "
        f"{p['calibrated_stim_with_idle']['z_ancilla_m1_one']:.3f} with the {p['calibrated_stim_with_idle']['idle_ns']:.0f} ns readout idle.",
        f"5. The detection fraction is in [{b['detection_fraction'][0]:.3f}, {b['detection_fraction'][1]:.3f}]: half the "
        + (f"stand-in's {si['detection_fraction']:.3f}" if si else f"no-idle prior's {p['calibrated_stim_no_idle']['detection_fraction']:.3f}")
        + f", up to twice the with-idle prior's {p['calibrated_stim_with_idle']['detection_fraction']:.3f}"
        f" (no-idle prior {p['calibrated_stim_no_idle']['detection_fraction']:.3f}).",
        "6. The job timeline gives the real per-shot overhead. Re-estimate the plan from it before E-B1.",
        "",
        "The full bands, with how each number is computed, are in `InfoRecord/emerald/circuits/job_plan.json: pilot`.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()
    cal = EM.live_calibration()
    suite = EM.build(cal)
    n = check_recorded(suite)
    jobs = EM.jobs(suite)
    plan = [SU.estimate(j, suite) for j in jobs]
    pilot = pilot_expectations(suite)
    per_shot = SU.per_shot_overhead_s(cal)
    pilot_us = C.schedule(suite[pilot["circuit"]].hardware, suite.durations).duration_ns / 1e3
    md = "\n".join([
        BEGIN,
        f"Generated by `scripts/emerald_plan.py` from calibration set `{cal.calibration_set_id}` ({n} recorded circuits, "
        "every `native_sha256` checked).",
        "",
        f"**Estimated QPU time of a job = Σ over its circuits of shots × (circuit duration + reset_wait) + circuits × "
        f"{SU.PER_CIRCUIT_OVERHEAD_S} s.** reset_wait is Emerald's own passive-reset wait, {per_shot * 1e6:.0f} µs "
        f"(`gates.reset_wait` in the snapshot; Garnet's is 298 µs). Circuit durations come from Emerald's calibrated "
        f"durations (prx {cal.median('prx_ns'):.0f} ns, CZ {cal.median('cz_ns', couplers=True):.0f} ns, measure "
        f"{cal.median('measure_ns'):.0f} ns median; each readout layer takes its slowest qubit's), with barriers as sync "
        f"points (`circuits.schedule`): the pilot circuit is {pilot_us:.2f} µs. The 'pessimistic' column charges "
        f"{SU.PESSIMISTIC_PER_SHOT_S * 1e3:.2f} ms per shot, Garnet's CLOPS bound (no Emerald CLOPS figure was checked).",
        "",
        plan_table(plan),
        "",
        split_table(plan),
        "",
        pilot_md(pilot),
        END,
    ])
    print(md)
    if args.no_write:
        return 0
    (EM.RECORD / "circuits" / "job_plan.json").write_text(json.dumps(EM.plan_json(suite, plan, pilot), indent=1, default=str) + "\n")
    if DOC.exists():
        text = DOC.read_text()
        if BEGIN in text and END in text:
            DOC.write_text(text[: text.index(BEGIN)] + md + text[text.index(END) + len(END):])
    return 0


if __name__ == "__main__":
    sys.exit(main())
