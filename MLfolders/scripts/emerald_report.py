"""The Emerald dry run's results and placements, as tables in docs/emerald-design.md (read-only on the record).

    uv run python scripts/emerald_report.py [--no-write]

Reads ``InfoRecord/emerald/sim/checks/*.json``, ``sim/ler/*/*/ler_fits.json`` and
``circuits/job_plan.json`` (``choices``), and replaces the text between ``<!-- report:begin -->``
and ``<!-- report:end -->`` in ``docs/emerald-design.md``. Every number is SIMULATED except the
placement ranking, which is calibrated Stim noise from the live calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from dqiqm.experiments import emerald as EM

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "emerald-design.md"
SIM = EM.RECORD / "sim"
BEGIN, END = "<!-- report:begin -->", "<!-- report:end -->"


def load(name: str):
    return json.loads((SIM / "checks" / name).read_text())


def ranking(plan) -> str:
    e1 = plan["choices"]["E1"]
    rows = defaultdict(list)
    for p in e1["placements"]:
        rows[p["qubit_set"]].append(p)
    lines = [f"Ranked by mean P_L at R = 3 (Z and X), PyMatching under calibrated Stim noise, {e1['rank_shots']:,} shots per basis "
             "and placement (`memory.rank` on the 85 calibrated couplers); one row per qubit set, its best orientation.",
             "", "| rank | patch | qubits | best orientation | X-type ancillas | P_L Z | P_L X | mean |", "|---|---|---|---|---|---|---|---|"]
    for i, qs in enumerate(e1["qubit_sets"]):
        best = min(rows[i], key=lambda p: p["mean_p_l_r3"])
        patch = next((p["patch"] for p in rows[i] if p["patch"]), "")
        lines.append(f"| {i + 1} | {patch} | {', '.join(qs)} | {best['orientation']} | {', '.join(best['x_ancillas'])} | "
                     f"{best['p_l_r3']['Z']:.4f} | {best['p_l_r3']['X']:.4f} | {best['mean_p_l_r3']:.4f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    plan = json.loads((EM.RECORD / "circuits" / "job_plan.json").read_text())
    summ, st, nl, dc, cc = (load(f) for f in ("summary.json", "structure.json", "noiseless.json", "decode.json", "crosscheck.json"))
    out = [BEGIN, "### The placements", "", ranking(plan), ""]
    out += ["### Structure (every circuit)", "",
            "| experiment | circuits | detectors | measurements | observables | SWAPs | CZ on calibrated couplers | DEM errors | graphlike distance |",
            "|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in st["rows"]:
        by[r["experiment"]].append(r)
    for exp, rs in sorted(by.items()):
        rng = lambda k: f"{min(r[k] for r in rs)}–{max(r[k] for r in rs)}"  # noqa: E731
        out.append(f"| {exp} | {len(rs)} | {rng('detectors')} | {rng('measurements')} | {sorted({r['observables'] for r in rs})} | "
                   f"{sorted({r['swaps'] for r in rs})} | all ({rng('cz')} per circuit) | {rng('dem_errors')} | "
                   f"{sorted({r['graphlike_distance'] for r in rs})} |")
    nl_rows = nl["rows"]
    out += ["", f"**Noiseless, through the hardware path** (`checks/noiseless.json`): all {len(nl_rows)} circuits, "
            f"{nl_rows[0]['shots']} shots each, Aer without noise → `hw.save_raw` → `decode.load_memory` → `detection_events`: "
            f"**{sum(r['detection_events'] for r in nl_rows)} detection events, {sum(r['observable_flips'] for r in nl_rows)} "
            f"observable flips**, with up to {max(r['random_columns'] for r in nl_rows)} random columns per circuit.", ""]
    out += [f"**Detectors against the team's `make_stim_circuit`** (`checks/crosscheck.json`, team-main `{cc.get('team_head', '')[:7]}`, "
            f"loaded read-only): {cc.get('circuits')} circuits (E1 and E4), detectors equal in order (record sets), "
            f"`get_detector_coordinates()` equal, observable equal: **pass {cc.get('pass')}**.", ""]
    p = dc["pooled"]
    out += [f"**E1 decoded against 'always 0'** (`checks/decode.json`; PyMatching on the noisy copy of the circuit that ran, "
            f"`emerald.calibrated_noise(..., idle_ns=0)`). Pooled: P_L = {p['p_pm']:.4f} [{p['p_pm_ci'][0]:.4f}, {p['p_pm_ci'][1]:.4f}] "
            f"decoded, against {p['p_raw']:.4f} [{p['p_raw_ci'][0]:.4f}, {p['p_raw_ci'][1]:.4f}] undecoded ({p['shots']:,} shots): "
            f"**better: {p['better_than_always_0']}**.", "",
            "| patch | basis | R | shots | P_L PyMatching [Wilson 95%] | P_L always 0 [Wilson 95%] |", "|---|---|---|---|---|---|"]
    for r in sorted(dc["rows"], key=lambda r: (r["patch"], r["basis"], r["rounds"])):
        out.append(f"| {r['patch']} | {r['basis']} | {r['rounds']} | {r['shots']} | {r['p_pm']:.4f} [{r['p_pm_ci'][0]:.4f}, {r['p_pm_ci'][1]:.4f}] | "
                   f"{r['p_raw']:.4f} [{r['p_raw_ci'][0]:.4f}, {r['p_raw_ci'][1]:.4f}] |")
    out += ["", "**LER records** (`sim/ler/<experiment>/<run id>/`, 2P = 1 − A(1 − 2ε)^R by maximum likelihood, "
            "`dqiqm.stats.fit_eps`, 95% bootstrap intervals; `ler_points.csv` has P_L per R with Wilson intervals). "
            "At these shot counts the intervals are wide: the point is the pipeline.", "",
            "| exp | run | ε_L (A = 1) [95%] | χ²/dof | ε_L (free A) [95%] | A | shots | rounds with 0 failures |",
            "|---|---|---|---|---|---|---|---|"]
    for f in sorted((SIM / "ler").glob("*/*/ler_fits.json")):
        b = json.loads(f.read_text())
        if "fixed_A" not in b:
            continue
        fa, fr = b["fixed_A"], b["free_A"]
        pts = (f.parent / "ler_points.csv").read_text().splitlines()[1:]
        shots = sum(int(dict(zip((f.parent / "ler_points.csv").read_text().splitlines()[0].split(","), x.split(",")))["shots"]) for x in pts)

        def ci(d):
            lo, hi = d.get("eps_ci_bootstrap") or (None, None)
            return f"[{lo:.4f}, {hi:.4f}]" if lo is not None and hi is not None else ""

        out.append(f"| {b['experiment']} | {b['run_id']} | {fa['eps']:.4f} {ci(fa)} | {fa['chi2_per_dof']:.2f} | {fr['eps']:.4f} {ci(fr)} | "
                   f"{fr['amplitude']:.3f} | {shots:,} | {b.get('zero_failure_points') or ''} |")
    dm = summ["derived_means"]
    out += ["", "**Derived features on the stand-in** (means over runs, `derived/*/*/summary.json`):", "",
            "| experiment | detection rate | same-stab lag 1 | same-stab lag 2 | cross-stab lag 0 |", "|---|---|---|---|---|"]
    for exp, v in dm.items():
        out.append(f"| {exp} | {v['det_rate_mean']:.4f} | {v['same_stab_lag1']:.4f} | {v['same_stab_lag2']:.4f} | {v['cross_stab_lag0']:.4f} |")
    out += ["", f"Dry run: {summ['jobs']} jobs, {summ['runs']} dataset runs, {summ['ler_records']} LER records, shots per circuit "
            f"{json.dumps(summ['dry_shots'])}; {summ['wall_s'] / 60:.1f} min wall on 4 cores under the sim lock.", END]
    md = "\n".join(out)
    print(md)
    if not args.no_write and DOC.exists():
        text = DOC.read_text()
        if BEGIN in text and END in text:
            DOC.write_text(text[: text.index(BEGIN)] + md + text[text.index(END) + len(END):])
    return 0


if __name__ == "__main__":
    sys.exit(main())
